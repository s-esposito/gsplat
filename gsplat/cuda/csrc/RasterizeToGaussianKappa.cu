/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>
#    include <cooperative_groups/reduce.h>

#    include "Common.h"
#    include "RasterizeToGaussians.h"
#    include "RasterizeToPixels3DGSDevice.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

constexpr uint32_t KAPPA_CHANNELS = 14;
constexpr uint32_t KAPPA_ROW      = 16; // a row's floats: the 14 channels, padded

// The traversal of rasterize_to_gaussians_kernel (same weights and cutoffs). Each (pixel, Gaussian i) pair,
// front to back, forms the contrast of i against what lies behind it at the pixel,
//   T_i (c_i - B_i) = T_i c_i - (C - C_<=i) / (1 - alpha_i)
// (C the rendered colour, C_<=i the colour accumulated up to and including i), then
//   w kappa = alpha_i < v, T_i (c_i - B_i) >,  v the per-pixel value (the loss gradient dL/dC, or the residual),
// and q, the point of the pixel's ray closest to i's centre in i's normalized frame S^-1 R^T (x - mean). Per
// Gaussian it accumulates 14 channels:
//   w v (3), w, w kappa, w kappa q (3), w kappa q_a q_b (xx, yy, zz, xy, xz, yz).
// Each thread handles P pixels in consecutive rows. Per Gaussian, each contributing pixel of a warp writes its 14
// values to a shared-memory row; lanes owning a channel sum it over the rows and add it to global memory.
template<uint32_t P>
__global__ void rasterize_to_gaussian_kappa_kernel(
    const uint32_t I,
    const int64_t n_isects,
    const vec2 *__restrict__ means2d,       // [..., C, N, 2]
    const vec3 *__restrict__ conics,        // [..., C, N, 3]
    const float *__restrict__ opacities,    // [..., C, N]
    const vec3 *__restrict__ colors,        // [..., C, N, 3]: each Gaussian's colour as rendered in that view
    const vec3 *__restrict__ render_colors, // [..., C, image_height, image_width, 3]
    const vec3 *__restrict__ pixel_values,  // [..., C, image_height, image_width, 3]
    const vec3 *__restrict__ means,         // [..., N, 3]
    const mat3 *__restrict__ frames,        // [..., N, 3, 3]: S^-1 R^T, row-major
    const float *__restrict__ viewmats,     // [..., C, 4, 4], world to camera
    const float *__restrict__ Ks,           // [..., C, 3, 3]
    const uint32_t N,
    const uint32_t n_cameras,
    const bool *__restrict__ masks, // [..., C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const int32_t *__restrict__ last_ids,     // [..., C, image_height, image_width]
    float *__restrict__ out                   // [..., C, N, 14]
)
{
    auto block              = cg::this_thread_block();
    const uint32_t image_id = block.group_index().x;
    const uint32_t tile_id  = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t j        = block.group_index().z * tile_size + block.thread_index().x;

    const int64_t tiles_per_image   = static_cast<int64_t>(tile_height) * tile_width;
    const int64_t pixels_per_image  = static_cast<int64_t>(image_height) * image_width;
    tile_offsets                   += image_id * tiles_per_image;
    last_ids                       += image_id * pixels_per_image;
    render_colors                  += image_id * pixels_per_image;
    pixel_values                   += image_id * pixels_per_image;
    if(masks != nullptr)
    {
        masks += image_id * tiles_per_image;
        if(!masks[tile_id])
        {
            return;
        }
    }

    // This image's camera: its centre and each pixel's ray direction, in world space.
    const float *vm = viewmats + image_id * 16;
    const float *K  = Ks + image_id * 9;
    vec3 cam_centre;
#    pragma unroll
    for(uint32_t a = 0; a < 3; ++a)
    {
        cam_centre[a] = -(vm[a] * vm[3] + vm[4 + a] * vm[7] + vm[8 + a] * vm[11]);
    }
    const int64_t gaussian_base = static_cast<int64_t>(image_id / n_cameras) * N; // this image's batch

    // The thread's P pixels, rows i0 .. i0 + P - 1. Out-of-image pixels hold 0 and never contribute.
    const uint32_t i0 = block.group_index().y * tile_size + block.thread_index().y * P;
    const float px    = static_cast<float>(j) + 0.5f;
    float py[P], T[P];
    bool inside[P];
    int32_t bin_final[P];
    vec3 v[P], C[P], acc[P], ray_world[P];
    int32_t max_bin_final = -1;
#    pragma unroll
    for(uint32_t p = 0; p < P; ++p)
    {
        const uint32_t i     = i0 + p;
        py[p]                = static_cast<float>(i) + 0.5f;
        inside[p]            = (i < image_height && j < image_width);
        const int64_t pix_id = static_cast<int64_t>(i) * image_width + j;
        bin_final[p]         = inside[p] ? last_ids[pix_id] : -1;
        max_bin_final        = max(max_bin_final, bin_final[p]);
        v[p]                 = inside[p] ? pixel_values[pix_id] : vec3(0.0f);
        C[p]                 = inside[p] ? render_colors[pix_id] : vec3(0.0f);
        acc[p]               = vec3(0.0f); // the colour accumulated so far, front to back
        T[p]                 = 1.0f;
        const vec3 ray_cam   = {(px - K[2]) / K[0], (py[p] - K[5]) / K[4], 1.0f};
#    pragma unroll
        for(uint32_t a = 0; a < 3; ++a)
        {
            ray_world[p][a] = vm[a] * ray_cam[0] + vm[4 + a] * ray_cam[1] + vm[8 + a] * ray_cam[2];
        }
    }

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const int64_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);         // [block_size]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t lane            = tr % 32;
    // Only tile_size 4 has a 16-lane warp: one lane per channel; otherwise two, which split the rows.
    const uint32_t lanes_per_ch    = block_size >= 32 ? 2 : 1;
    const uint32_t channel         = lane / lanes_per_ch;
    const uint32_t part            = lane % lanes_per_ch;
    const int32_t warp_bin_final   = cg::reduce(warp, max_bin_final, cg::greater<int>());
    // This warp's rows, one per contributing pixel (up to 32 P).
    float *rows = reinterpret_cast<float *>(&conic_batch[block_size]) + (tr / 32) * (32 * P * KAPPA_ROW);

    for(int64_t b = 0; b < num_batches; ++b)
    {
        const int64_t batch_offset = static_cast<int64_t>(block_size) * b;
        if(__syncthreads_count(max_bin_final < batch_offset) >= block_size)
        {
            break;
        }

        const int64_t idx = range_start + batch_offset + tr;
        if(idx < range_end)
        {
            const int32_t g      = flatten_ids[idx];
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
            conic_batch[tr]      = conics[g];
        }
        block.sync();

        const int64_t remaining      = range_end - range_start - batch_offset;
        const uint32_t batch_size    = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        // Bounded by the warp's furthest contributor so every lane runs the same iterations (the ballots and the
        // row sums below are warp collectives).
        const int64_t warp_remaining = static_cast<int64_t>(warp_bin_final) - batch_offset + 1;
        const uint32_t end_t = warp_remaining <= 0
                                 ? 0
                                 : static_cast<uint32_t>(warp_remaining < batch_size ? warp_remaining : batch_size);
        for(uint32_t t = 0; t < end_t; ++t)
        {
            bool valid[P];
            float alpha[P];
            bool any_valid     = false;
            const vec3 xy_opac = xy_opacity_batch[t];
            const vec3 conic   = conic_batch[t];
#    pragma unroll
            for(uint32_t p = 0; p < P; ++p)
            {
                valid[p] = inside[p] && (batch_offset + t <= bin_final[p]);
                alpha[p] = 0.0f;
                if(valid[p])
                {
                    const GaussianWeight gw = eval_gaussian_weight(conic, xy_opac.x - px, xy_opac.y - py[p], xy_opac.z);
                    valid[p]                = gw.valid;
                    alpha[p]                = gw.alpha;
                }
                any_valid = any_valid || valid[p];
            }
            if(!warp.any(any_valid))
            {
                continue;
            }

            // Contributing pixels fill rows 0 .. n - 1: first every lane's pixel 0, then pixel 1, ...
            uint32_t n_rows = 0, my_row[P];
#    pragma unroll
            for(uint32_t p = 0; p < P; ++p)
            {
                const uint32_t ballot  = warp.ballot(valid[p]);
                my_row[p]              = n_rows + __popc(ballot & ((1u << lane) - 1));
                n_rows                += __popc(ballot);
            }
            const int64_t g = id_batch[t]; // flatten index in [I * N]
            if(any_valid)
            {
                // Every lane of the warp reads the same Gaussian (one cached load each).
                const vec3 c     = colors[g];
                const int64_t gr = gaussian_base + g % N;
                const mat3 M_    = frames[gr]; // row-major: M_[a] is row a
                const vec3 rel   = cam_centre - means[gr];
                vec3 o;
#    pragma unroll
                for(uint32_t a = 0; a < 3; ++a)
                {
                    o[a] = glm::dot(M_[a], rel);
                }
#    pragma unroll
                for(uint32_t p = 0; p < P; ++p)
                {
                    if(!valid[p])
                    {
                        continue;
                    }
                    const float w          = alpha[p] * T[p];
                    const vec3 acc_incl    = acc[p] + w * c;
                    // T (c - B), from C = acc_incl + T (1 - alpha) B; alpha <= 0.99 keeps the division bounded.
                    const vec3 contrast_T  = T[p] * c - (C[p] - acc_incl) / (1.0f - alpha[p]);
                    const float wk         = alpha[p] * glm::dot(v[p], contrast_T); // w * kappa
                    T[p]                  *= 1.0f - alpha[p];
                    acc[p]                 = acc_incl;

                    // q = d x (o x d) / (d . d), the ray o + t d in the normalized frame (as
                    // rasterize_to_gaussian_grids: nothing large cancels for flat Gaussians).
                    vec3 d;
#    pragma unroll
                    for(uint32_t a = 0; a < 3; ++a)
                    {
                        d[a] = glm::dot(M_[a], ray_world[p]);
                    }
                    const vec3 q = glm::cross(d, glm::cross(o, d)) / glm::dot(d, d);

                    float4 *row = reinterpret_cast<float4 *>(rows + my_row[p] * KAPPA_ROW);
                    row[0]      = {w * v[p].x, w * v[p].y, w * v[p].z, w};
                    row[1]      = {wk, wk * q.x, wk * q.y, wk * q.z};
                    row[2]      = {wk * q.x * q.x, wk * q.y * q.y, wk * q.z * q.z, wk * q.x * q.y};
                    row[3]      = {wk * q.x * q.z, wk * q.y * q.z, 0.0f, 0.0f};
                }
            }
            warp.sync();

            // Lanes owning a channel sum it over the rows (a strided walk shared by lanes_per_ch lanes, then one
            // shuffle) and add it to global memory.
            float sum = 0.0f;
            if(channel < KAPPA_CHANNELS)
            {
                for(uint32_t m = part; m < n_rows; m += lanes_per_ch)
                {
                    sum += rows[m * KAPPA_ROW + channel];
                }
            }
            if(lanes_per_ch == 2)
            {
                sum += warp.shfl_xor(sum, 1);
            }
            if(channel < KAPPA_CHANNELS && part == 0 && sum != 0.0f)
            {
                atomicAdd_system(out + g * KAPPA_CHANNELS + channel, sum);
            }
            warp.sync(); // the rows are rewritten for the next Gaussian
        }
    }
}

void launch_rasterize_to_gaussian_kappa_kernel(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    const at::Tensor colors,
    const at::Tensor render_colors,
    const at::Tensor pixel_values,
    const at::Tensor means,
    const at::Tensor frames,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor last_ids,
    at::Tensor out
)
{
    const int64_t n_isects = flatten_ids.size(0);
    if(n_isects == 0 || last_ids.numel() == 0)
    {
        return;
    }
    const uint32_t I           = last_ids.numel() / (static_cast<int64_t>(image_height) * image_width);
    const uint32_t tile_height = tile_offsets.size(-2);
    const uint32_t tile_width  = tile_offsets.size(-1);
    const dim3 grid            = {I, tile_height, tile_width};

    // 2 pixels per thread for 16 x 16 tiles (a warp covers 16 x 4 pixels, so each Gaussian costs half as many warp
    // passes); tile_size 4 keeps 1 (its 16-thread block would otherwise be 8 threads).
    auto launch = [&]<uint32_t P>()
    {
        const dim3 threads       = {tile_size, tile_size / P, 1};
        const uint32_t n_threads = tile_size * tile_size / P;
        const uint32_t n_warps   = (n_threads + 31) / 32;
        const int64_t shmem_size
            = n_threads * (sizeof(int32_t) + 2 * sizeof(vec3)) + n_warps * 32 * P * KAPPA_ROW * sizeof(float);
        if(cudaFuncSetAttribute(
               rasterize_to_gaussian_kappa_kernel<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }
        rasterize_to_gaussian_kappa_kernel<P><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            n_isects,
            reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
            opacities.const_data_ptr<float>(),
            reinterpret_cast<const vec3 *>(colors.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(render_colors.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(pixel_values.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(means.const_data_ptr<float>()),
            reinterpret_cast<const mat3 *>(frames.const_data_ptr<float>()),
            viewmats.const_data_ptr<float>(),
            Ks.const_data_ptr<float>(),
            static_cast<uint32_t>(means.size(-2)),
            static_cast<uint32_t>(viewmats.size(-3)),
            masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.const_data_ptr<int64_t>(),
            flatten_ids.const_data_ptr<int32_t>(),
            last_ids.const_data_ptr<int32_t>(),
            out.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if(tile_size == 16)
    {
        launch.template operator()<2>();
    }
    else
    {
        launch.template operator()<1>();
    }
}
} // namespace gsplat

#endif
