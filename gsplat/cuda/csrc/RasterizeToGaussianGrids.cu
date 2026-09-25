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

constexpr uint32_t GRID_CELLS = 27;

// The traversal of rasterize_to_gaussians_kernel (same weights and cutoffs), but
// each (pixel, Gaussian) pair splats w * pixel value trilinearly into a 3x3x3
// grid in the Gaussian's normalized frame, at q = the point of the pixel's ray
// closest to the centre. Each thread handles P pixels in consecutive rows (a
// warp covers a 16 x 2P patch). Per Gaussian, each contributing pixel writes its
// trilinear weights and weighted values to a shared-memory row; lanes owning a
// cell sum over the rows and add the cell to global memory.
template<uint32_t CDIM, uint32_t P>
__global__ void rasterize_to_gaussian_grids_kernel(
    const uint32_t I,
    const int64_t n_isects,
    const vec2 *__restrict__ means2d,       // [..., C, N, 2]
    const vec3 *__restrict__ conics,        // [..., C, N, 3]
    const float *__restrict__ opacities,    // [..., C, N]
    const float *__restrict__ pixel_values, // [..., C, image_height, image_width, CDIM]
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
    float *__restrict__ out_values,           // [..., C, N, 27, CDIM]
    float *__restrict__ out_weights           // [..., C, N, 27]
)
{
    constexpr uint32_t C    = CDIM + 1;        // values, then the weight
    constexpr uint32_t VS   = (C + 3) / 4 * 4; // values per row, padded
    constexpr uint32_t ROW  = 12 + VS;         // x_i * y_j (9), z_k (3), values
    auto block              = cg::this_thread_block();
    const uint32_t image_id = block.group_index().x;
    const uint32_t tile_id  = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t j        = block.group_index().z * tile_size + block.thread_index().x;

    const int64_t tiles_per_image   = static_cast<int64_t>(tile_height) * tile_width;
    const int64_t pixels_per_image  = static_cast<int64_t>(image_height) * image_width;
    tile_offsets                   += image_id * tiles_per_image;
    last_ids                       += image_id * pixels_per_image;
    pixel_values                   += image_id * pixels_per_image * CDIM;
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

    // The thread's P pixels, rows i0 .. i0 + P - 1.
    const uint32_t i0 = block.group_index().y * tile_size + block.thread_index().y * P;
    float px, py[P], T[P], pix_value[P][CDIM];
    bool inside[P];
    int32_t bin_final[P];
    vec3 ray_world[P];
    px                    = static_cast<float>(j) + 0.5f;
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
        T[p]                 = 1.0f;
#    pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            pix_value[p][k] = inside[p] ? pixel_values[pix_id * CDIM + k] : 0.0f;
        }
        const vec3 ray_cam = {(px - K[2]) / K[0], (py[p] - K[5]) / K[4], 1.0f};
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
    // Only tile_size 4 has a 16-lane warp.
    const uint32_t warp_lanes      = block_size < 32 ? block_size : 32;
    const int32_t warp_bin_final   = cg::reduce(warp, max_bin_final, cg::greater<int>());
    // This warp's rows, one per contributing pixel (up to 32 P): the trilinear weights factored as x_i * y_j (9) and
    // z_k (3), then its C weighted values (padded to VS for float4 loads).
    float *rows                    = reinterpret_cast<float *>(&conic_batch[block_size]) + (tr / 32) * (32 * P * ROW);

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
        const int64_t warp_remaining = static_cast<int64_t>(warp_bin_final) - batch_offset + 1;
        const uint32_t end_t = warp_remaining <= 0
                                 ? 0
                                 : static_cast<uint32_t>(warp_remaining < batch_size ? warp_remaining : batch_size);
        for(uint32_t t = 0; t < end_t; ++t)
        {
            bool valid[P];
            float w[P];
            bool any_valid     = false;
            const vec3 xy_opac = xy_opacity_batch[t];
            const vec3 conic   = conic_batch[t];
#    pragma unroll
            for(uint32_t p = 0; p < P; ++p)
            {
                valid[p] = inside[p] && (batch_offset + t <= bin_final[p]);
                w[p]     = 0.0f;
                if(valid[p])
                {
                    const GaussianWeight gw = eval_gaussian_weight(conic, xy_opac.x - px, xy_opac.y - py[p], xy_opac.z);
                    if(gw.valid)
                    {
                        w[p] = gw.alpha * T[p];
                        T[p] = T[p] * (1.0f - gw.alpha);
                    }
                    else
                    {
                        valid[p] = false;
                    }
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
            uint32_t cell_mask = 0; // the cells this lane's pixels reach with a nonzero weight, bit 9 i + 3 j + k
            if(any_valid)
            {
                // The ray in the Gaussian's normalized frame x' = M (x - mean), M = S^-1 R^T: origin o, direction
                // d. M and the mean are read from global memory: every lane of the warp reads the same Gaussian's
                // (one cached load). M is row-major and glm's mat3 column-major, so M_[a] is row a.
                const int64_t gr = gaussian_base + id_batch[t] % N;
                const mat3 M_    = frames[gr];
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
                    vec3 d;
#    pragma unroll
                    for(uint32_t a = 0; a < 3; ++a)
                    {
                        d[a] = glm::dot(M_[a], ray_world[p]);
                    }
                    // q = o + t d with t = -(o.d)/(d.d), written as d x (o x d) / (d.d): for a flat Gaussian o and
                    // d are huge along its thin axis, and o + t d would cancel two such terms (float32 noise of ~1
                    // sigma); each component of o x d pairs two different axes, so nothing large cancels.
                    const vec3 q = glm::cross(d, glm::cross(o, d)) / glm::dot(d, d);
                    // Per axis, cell centres at -1, 0, +1: weights 1 - f and f on the two cells around q.
                    float axis_w[3][3];
#    pragma unroll
                    for(uint32_t a = 0; a < 3; ++a)
                    {
                        const float u  = fminf(fmaxf(q[a] + 1.0f, 0.0f), 2.0f);
                        const float lo = u >= 1.0f ? 1.0f : 0.0f;
                        const float f  = u - lo;
                        axis_w[a][0]   = lo == 0.0f ? 1.0f - f : 0.0f;
                        axis_w[a][1]   = lo == 0.0f ? f : 1.0f - f;
                        axis_w[a][2]   = lo == 0.0f ? 0.0f : f;
                    }
                    // The cells with a nonzero weight: those whose index along every axis has one. Bit 9 i + 3 j + k
                    // of the AND of three per-axis patterns (all cells with i = 0 are bits 0..8, and so on).
                    uint32_t along[3] = {0, 0, 0};
#    pragma unroll
                    for(uint32_t c = 0; c < 3; ++c)
                    {
                        along[0] |= axis_w[0][c] != 0.0f ? 0x1FFu << (9 * c) : 0u;
                        along[1] |= axis_w[1][c] != 0.0f ? 0x1C0E07u << (3 * c) : 0u;
                        along[2] |= axis_w[2][c] != 0.0f ? 0x1249249u << c : 0u;
                    }
                    cell_mask  |= along[0] & along[1] & along[2];
                    float *row  = rows + my_row[p] * ROW;
#    pragma unroll
                    for(uint32_t a = 0; a < 3; ++a)
                    {
#    pragma unroll
                        for(uint32_t b2 = 0; b2 < 3; ++b2)
                        {
                            row[a * 3 + b2] = axis_w[0][a] * axis_w[1][b2];
                        }
                        row[9 + a] = axis_w[2][a];
                    }
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        row[12 + k] = w[p] * pix_value[p][k];
                    }
                    row[12 + CDIM] = w[p];
                }
            }
            warp.sync();

            // Only the cells some pixel reaches are summed (a warp covers a small patch of the Gaussian). Each gets
            // lanes_per_cell lanes (the largest power of two that fits), which split its rows and then add their
            // partial sums with shuffles. Every lane of a group reads the same row at once (a shared-memory
            // broadcast), and there are no atomics until global memory.
            const uint32_t active   = cg::reduce(warp, cell_mask, cg::bit_or<uint32_t>());
            const uint32_t n_active = __popc(active);
            uint32_t lanes_per_cell = 1;
            while(lanes_per_cell * 2 * n_active <= warp_lanes)
            {
                lanes_per_cell *= 2;
            }
            const uint32_t n_groups = warp_lanes / lanes_per_cell;
            const uint32_t part     = lane % lanes_per_cell;
            const int64_t g         = id_batch[t];
            // Sum cell `slot` (the slot-th active one) over the rows and add it to global memory; every lane calls it.
            auto sum_cell           = [&](const uint32_t slot)
            {
                const bool owner  = slot < n_active;
                const uint32_t v  = owner ? __fns(active, 0, slot + 1) : 0;
                const uint32_t ci = v / 9, cj = v / 3 % 3, ck = v % 3;
                float acc[C];
#    pragma unroll
                for(uint32_t k = 0; k < C; ++k)
                {
                    acc[k] = 0.0f;
                }
                if(owner)
                {
#    pragma unroll 4
                    for(uint32_t m = part; m < n_rows; m += lanes_per_cell)
                    {
                        const float *row = rows + m * ROW;
                        const float tri  = row[ci * 3 + cj] * row[9 + ck];
                        const float4 *v4 = reinterpret_cast<const float4 *>(row + 12);
#    pragma unroll
                        for(uint32_t k4 = 0; k4 < VS / 4; ++k4)
                        {
                            const float4 val = v4[k4];
                            const float x[4] = {val.x, val.y, val.z, val.w};
#    pragma unroll
                            for(uint32_t e = 0; e < 4; ++e)
                            {
                                if(k4 * 4 + e < C)
                                {
                                    acc[k4 * 4 + e] += tri * x[e];
                                }
                            }
                        }
                    }
                }
                for(uint32_t offset = lanes_per_cell / 2; offset > 0; offset /= 2)
                {
#    pragma unroll
                    for(uint32_t k = 0; k < C; ++k)
                    {
                        acc[k] += warp.shfl_xor(acc[k], offset);
                    }
                }
                if(owner && part == 0 && acc[CDIM] != 0.0f)
                {
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        atomicAdd_system(out_values + (g * GRID_CELLS + v) * CDIM + k, acc[k]);
                    }
                    atomicAdd_system(out_weights + g * GRID_CELLS + v, acc[CDIM]);
                }
            };
            sum_cell(lane / lanes_per_cell);
            // Only a 16-lane warp (tile_size 4) can have more active cells than lanes: a second round takes the rest.
            if(n_active > n_groups)
            {
                sum_cell(lane / lanes_per_cell + n_groups);
            }
            warp.sync(); // the rows are rewritten for the next Gaussian
        }
    }
}

void launch_rasterize_to_gaussian_grids_kernel(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
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
    at::Tensor out_values,
    at::Tensor out_weights
)
{
    const int64_t n_isects = flatten_ids.size(0);
    if(n_isects == 0 || last_ids.numel() == 0)
    {
        return;
    }
    const uint32_t I           = last_ids.numel() / (static_cast<int64_t>(image_height) * image_width);
    const uint32_t D           = pixel_values.size(-1);
    const uint32_t tile_height = tile_offsets.size(-2);
    const uint32_t tile_width  = tile_offsets.size(-1);

    const dim3 grid = {I, tile_height, tile_width};

    // 2 pixels per thread for 16 x 16 tiles (a warp covers 16 x 4 pixels, so each Gaussian costs half as many
    // warp passes); tile_size 4 keeps 1 (its 16-thread block would otherwise be 8 threads).
    auto launch = [&]<uint32_t CDIM, uint32_t P>()
    {
        constexpr uint32_t C     = CDIM + 1;
        const dim3 threads       = {tile_size, tile_size / P, 1};
        const uint32_t n_threads = tile_size * tile_size / P;
        const uint32_t n_warps   = (n_threads + 31) / 32;
        const int64_t shmem_size = n_threads * (sizeof(int32_t) + 2 * sizeof(vec3))
                                 + n_warps * 32 * P * (12 + (C + 3) / 4 * 4) * sizeof(float);
        if(cudaFuncSetAttribute(
               rasterize_to_gaussian_grids_kernel<CDIM, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }
        // The largest shared-memory share of L1, so that more blocks fit per SM.
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            rasterize_to_gaussian_grids_kernel<CDIM, P>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        ));
        rasterize_to_gaussian_grids_kernel<CDIM, P><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            n_isects,
            reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
            opacities.const_data_ptr<float>(),
            pixel_values.const_data_ptr<float>(),
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
            out_values.data_ptr<float>(),
            out_weights.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    auto launch_p = [&]<uint32_t CDIM>()
    {
        if(tile_size == 16)
        {
            launch.template operator()<CDIM, 2>();
        }
        else
        {
            launch.template operator()<CDIM, 1>();
        }
    };
    switch(D)
    {
    case 1:  launch_p.template operator()<1>(); break;
    case 2:  launch_p.template operator()<2>(); break;
    case 3:  launch_p.template operator()<3>(); break;
    case 4:  launch_p.template operator()<4>(); break;
    case 5:  launch_p.template operator()<5>(); break;
    case 6:  launch_p.template operator()<6>(); break;
    case 7:  launch_p.template operator()<7>(); break;
    case 8:  launch_p.template operator()<8>(); break;
    default: AT_ERROR("rasterize_to_gaussian_grids supports 1 to 8 channels, got ", D);
    }
}
} // namespace gsplat

#endif
