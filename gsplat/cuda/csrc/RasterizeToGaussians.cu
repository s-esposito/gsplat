/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/core/Tensor.h>
#    include <ATen/cuda/Atomic.cuh>
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

// Front-to-back traversal mirroring RasterizeToPixels3DGSSerialBatchFwd (same
// weight and transmittance expressions), accumulating per Gaussian with the
// warp-reduce + rank-0 atomic add pattern of RasterizeToPixels3DGSSerialBatchBwd.
__global__ void rasterize_to_gaussians_kernel(
    const uint32_t I,
    const uint32_t D,
    const int64_t n_isects,
    const vec2 *__restrict__ means2d,       // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,        // [..., N, 3] or [nnz, 3]
    const float *__restrict__ opacities,    // [..., N] or [nnz]
    const float *__restrict__ pixel_values, // [..., image_height, image_width, D]
    const bool *__restrict__ masks,         // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const int32_t *__restrict__ last_ids,     // [..., image_height, image_width]
    float *__restrict__ out_values,           // [..., N, D] or [nnz, D]
    float *__restrict__ out_weights           // [..., N] or [nnz]
)
{
    auto block              = cg::this_thread_block();
    const uint32_t image_id = block.group_index().x;
    const uint32_t tile_id  = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i        = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j        = block.group_index().z * tile_size + block.thread_index().x;

    const int64_t tiles_per_image   = static_cast<int64_t>(tile_height) * tile_width;
    const int64_t pixels_per_image  = static_cast<int64_t>(image_height) * image_width;
    tile_offsets                   += image_id * tiles_per_image;
    last_ids                       += image_id * pixels_per_image;
    pixel_values                   += image_id * pixels_per_image * D;
    if(masks != nullptr)
    {
        masks += image_id * tiles_per_image;
        // The forward skipped this tile; every thread of the block returns
        // before any collective.
        if(!masks[tile_id])
        {
            return;
        }
    }

    const float px         = static_cast<float>(j) + 0.5f;
    const float py         = static_cast<float>(i) + 0.5f;
    const bool inside      = (i < image_height && j < image_width);
    const int64_t pix_id   = static_cast<int64_t>(i) * image_width + j;
    const int64_t pix_base = pix_id * D;
    // Out-of-image threads keep loading shared memory but never contribute.
    const int32_t bin_final = inside ? last_ids[pix_id] : -1;

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
    const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());
    float T                        = 1.0f;

    for(int64_t b = 0; b < num_batches; ++b)
    {
        // Tile-relative offset of the first gaussian in this batch.
        const int64_t batch_offset = static_cast<int64_t>(block_size) * b;

        // Resync the block before loading; stop once no pixel of the tile has a
        // contributor at or past this batch.
        if(__syncthreads_count(bin_final < batch_offset) >= block_size)
        {
            break;
        }

        const int64_t idx = range_start + batch_offset + tr;
        if(idx < range_end)
        {
            const int32_t g      = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            const float opac     = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr]      = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        const int64_t remaining   = range_end - range_start - batch_offset;
        const uint32_t batch_size = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        // Bounded by the warp's furthest contributor so every lane of the warp
        // runs the same iterations (the reductions below are collectives).
        const int64_t warp_remaining = static_cast<int64_t>(warp_bin_final) - batch_offset + 1;
        const uint32_t end_t
            = warp_remaining <= 0 ? 0 : static_cast<uint32_t>(warp_remaining < batch_size ? warp_remaining : batch_size);
        for(uint32_t t = 0; t < end_t; ++t)
        {
            bool valid = inside && (batch_offset + t <= bin_final);
            float w    = 0.0f;
            if(valid)
            {
                const vec3 conic        = conic_batch[t];
                const vec3 xy_opac      = xy_opacity_batch[t];
                const float opac        = xy_opac.z;
                const float dx          = xy_opac.x - px;
                const float dy          = xy_opac.y - py;
                const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
                if(gw.valid)
                {
                    const float alpha  = gw.alpha;
                    const float next_T = T * (1.0f - alpha);
                    w                  = alpha * T;
                    T                  = next_T;
                }
                else
                {
                    valid = false;
                }
            }

            if(!warp.any(valid))
            {
                continue;
            }

            const bool rank0  = warp.thread_rank() == 0;
            const int64_t g   = id_batch[t]; // flatten index in [I * N] or [nnz]
            const float w_sum = cg::reduce(warp, w, cg::plus<float>());
            if(rank0)
            {
                atomicAdd_system(out_weights + g, w_sum);
            }
            for(uint32_t k = 0; k < D; ++k)
            {
                // Invalid lanes must not read pixel_values: out-of-image lanes
                // would read out of bounds, and 0 * NaN would poison the sum.
                float v = valid ? w * pixel_values[pix_base + k] : 0.0f;
                v       = cg::reduce(warp, v, cg::plus<float>());
                if(rank0)
                {
                    atomicAdd_system(out_values + g * D + k, v);
                }
            }
        }
    }
}

void launch_rasterize_to_gaussians_kernel(
    // Gaussian parameters
    const at::Tensor means2d,             // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,              // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,           // [..., N] or [nnz]
    const at::Tensor pixel_values,        // [..., image_height, image_width, D]
    const at::optional<at::Tensor> masks, // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections and forward outputs
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const at::Tensor last_ids,     // [..., image_height, image_width]
    // outputs
    at::Tensor out_values, // [..., N, D] or [nnz, D]
    at::Tensor out_weights // [..., N] or [nnz]
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

    // Each block covers a tile on the image; one thread per pixel.
    const dim3 threads = {tile_size, tile_size, 1};
    const dim3 grid    = {I, tile_height, tile_width};

    const int64_t shmem_size = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));
    if(cudaFuncSetAttribute(rasterize_to_gaussians_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size)
       != cudaSuccess)
    {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size.");
    }

    rasterize_to_gaussians_kernel<<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
        I,
        D,
        n_isects,
        reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
        reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
        opacities.const_data_ptr<float>(),
        pixel_values.const_data_ptr<float>(),
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
}
} // namespace gsplat

#endif
