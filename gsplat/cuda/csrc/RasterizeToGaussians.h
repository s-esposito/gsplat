/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <ATen/core/Tensor.h>

namespace gsplat
{
// Accumulates per-pixel values onto the Gaussians that blended into each pixel,
// weighted by the forward blending weight alpha * T. The contributor set is
// bounded by the forward's last_ids.
void launch_rasterize_to_gaussians_kernel(
    // Gaussian parameters (as fed to the forward rasterizer)
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
);
} // namespace gsplat
