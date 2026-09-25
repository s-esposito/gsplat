# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rasterize_to_gaussians``.

The op accumulates per-pixel values onto the Gaussians blended into each pixel,
weighted by the forward blending weight w = alpha * T. It is validated against
three independent references:

- the per-pixel contributor ids/weights from ``rasterize_contributing_gaussian_ids``
  scattered onto Gaussians (same front-to-back math),
- the vector-Jacobian product of ``rasterize_to_pixels`` w.r.t. colors (the
  backward kernel, which reconstructs T back to front),
- the invariant that each image's accumulated weights sum to its alpha sum.
"""

import math

import pytest
import torch

import gsplat
from gsplat._helper import load_test_data
from gsplat.cuda._wrapper import (
    _make_lazy_cuda_func,
    isect_offset_encode,
    isect_tiles,
    rasterize_contributing_gaussian_ids,
    rasterize_num_contributing_gaussians,
    rasterize_to_gaussian_grids,
    rasterize_to_gaussians,
    rasterize_to_pixels,
)

device = torch.device("cuda:0")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device"),
    pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in"),
]

# Odd sizes so the last row/column of tiles is partial.
WIDTH, HEIGHT = 37, 29


def _make_scene(
    batch_dims,
    C,
    N,
    tile_size,
    packed,
    width=WIDTH,
    height=HEIGHT,
    seed=0,
    opacity_range=(0.05, 0.95),
):
    """Anisotropic 2D Gaussians with their tile intersections.

    Returns the forward's inputs (means2d, conics, opacities: dense [..., C, N, *]
    or packed [nnz, *]; isect_offsets; flatten_ids; width; height; tile_size;
    packed) plus image_ids (flat image index per Gaussian row), I and N.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*s, device=device, generator=gen)
    I = math.prod(batch_dims) * C

    means2d = torch.stack([u(I, N) * width, u(I, N) * height], dim=-1)
    sx = u(I, N) * 4.0 + 1.0
    sy = u(I, N) * 4.0 + 1.0
    rho = (u(I, N) * 2.0 - 1.0) * 0.8
    cxy = rho * sx * sy
    det = (sx * sy) ** 2 * (1.0 - rho**2)
    conics = torch.stack([sy**2 / det, -cxy / det, sx**2 / det], dim=-1)
    r = (3.0 * torch.maximum(sx, sy)).ceil().to(torch.int32)
    radii = torch.stack([r, r], dim=-1)
    depths = u(I, N) * 9.9 + 0.1
    lo, hi = opacity_range
    opacities = u(I, N) * (hi - lo) + lo

    th, tw = math.ceil(height / tile_size), math.ceil(width / tile_size)
    if packed:
        keep = u(I, N) > 0.3  # nnz is not a multiple of N
        image_ids, gaussian_ids = torch.where(keep)
        means2d, conics, opacities = means2d[keep], conics[keep], opacities[keep]
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii[keep],
            depths[keep],
            tile_size,
            tw,
            th,
            packed=True,
            n_images=I,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        shape = batch_dims + (C, N)
        means2d = means2d.reshape(shape + (2,))
        conics = conics.reshape(shape + (3,))
        opacities = opacities.reshape(shape)
        image_ids = torch.arange(I, device=device).repeat_interleave(N)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii.reshape(shape + (2,)),
            depths.reshape(shape),
            tile_size,
            tw,
            th,
        )
    isect_offsets = isect_offset_encode(isect_ids, I, tw, th).reshape(
        batch_dims + (C, th, tw)
    )
    return dict(
        means2d=means2d.contiguous(),
        conics=conics.contiguous(),
        opacities=opacities.contiguous(),
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
        tile_size=tile_size,
        packed=packed,
        image_ids=image_ids,
        I=I,
        N=N,
    )


def _forward(scene, masks=None, channels=3):
    """Forward render; returns (render_alphas, last_ids)."""
    colors = torch.zeros(scene["opacities"].shape + (channels,), device=device)
    _, alphas, _, last_ids = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
        scene["means2d"],
        scene["conics"],
        colors,
        scene["opacities"],
        None,
        masks,
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        scene["packed"],
        False,
    )
    return alphas, last_ids


def _to_gaussians(scene, pixel_values, last_ids, masks=None):
    return rasterize_to_gaussians(
        scene["means2d"],
        scene["conics"],
        scene["opacities"],
        pixel_values,
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        last_ids,
        masks=masks,
    )


def _vjp_reference(scene, pixel_values, masks=None):
    """sum_p w * pixel_values via the backward of rasterize_to_pixels w.r.t. colors."""
    colors = torch.zeros(
        scene["opacities"].shape + (pixel_values.shape[-1],),
        device=device,
        requires_grad=True,
    )
    render_colors, _ = rasterize_to_pixels(
        scene["means2d"],
        scene["conics"],
        colors,
        scene["opacities"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        masks=masks,
        packed=scene["packed"],
    )
    (v_colors,) = torch.autograd.grad((render_colors * pixel_values).sum(), colors)
    return v_colors


@pytest.mark.parametrize("D", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_matches_contributing_ids(tile_size, packed, batch_dims, D):
    C, N = 2, 60
    scene = _make_scene(batch_dims, C, N, tile_size, packed)
    I = scene["I"]
    _, last_ids = _forward(scene)
    pixel_values = torch.rand(batch_dims + (C, HEIGHT, WIDTH, D), device=device)
    values, weights = _to_gaussians(scene, pixel_values, last_ids)

    contrib_args = (
        scene["means2d"],
        scene["conics"],
        scene["opacities"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        WIDTH,
        HEIGHT,
        tile_size,
    )
    ncg, _ = rasterize_num_contributing_gaussians(*contrib_args)
    ids, contrib_w = rasterize_contributing_gaussian_ids(*contrib_args, ncg)
    K = ids.shape[-1]
    assert K > 0
    ids = ids.reshape(I, HEIGHT * WIDTH, K)
    contrib_w = contrib_w.reshape(I, HEIGHT * WIDTH, K)
    valid = ids >= 0
    # Dense ids are per-image local (g % N); packed ids index the nnz rows.
    if packed:
        rows = ids[valid]
        n_rows = scene["opacities"].shape[0]
    else:
        img = torch.arange(I, device=device)[:, None, None].expand_as(ids)
        rows = (img * N + ids)[valid]
        n_rows = I * N
    per_contrib = (contrib_w[..., None] * pixel_values.reshape(I, -1, 1, D))[valid]
    ref_values = torch.zeros(n_rows, D, device=device).index_add_(0, rows, per_contrib)
    ref_weights = torch.zeros(n_rows, device=device)
    ref_weights.index_add_(0, rows, contrib_w[valid])

    assert values.shape == scene["opacities"].shape + (D,)
    assert weights.shape == scene["opacities"].shape
    torch.testing.assert_close(
        values.reshape(n_rows, D), ref_values, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        weights.reshape(n_rows), ref_weights, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("use_masks", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (1, 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_matches_vjp(tile_size, packed, batch_dims, use_masks):
    C, N, D = 2, 60, 3
    scene = _make_scene(batch_dims, C, N, tile_size, packed, seed=1)
    masks = None
    if use_masks:
        gen = torch.Generator(device=device).manual_seed(7)
        shape = scene["isect_offsets"].shape
        masks = torch.rand(shape, device=device, generator=gen) > 0.3
    _, last_ids = _forward(scene, masks)
    pixel_values = torch.randn(batch_dims + (C, HEIGHT, WIDTH, D), device=device)

    values, weights = _to_gaussians(scene, pixel_values, last_ids, masks)
    ref_values = _vjp_reference(scene, pixel_values, masks)
    ones = torch.ones_like(pixel_values[..., :1])
    ref_weights = _vjp_reference(scene, ones, masks)[..., 0]

    torch.testing.assert_close(values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tile_size", [4, 16])
def test_saturation_and_multiple_batches(tile_size):
    # Many low-opacity Gaussians: tiles hold more intersections than one thread
    # block, contributors reach past the first batch, and pixels saturate (the
    # transmittance cutoff stops traversal).
    C, N, D, W, H = 1, 4000, 3, 64, 64
    scene = _make_scene(
        (), C, N, tile_size, False, W, H, seed=2, opacity_range=(0.05, 0.3)
    )
    block_size = tile_size * tile_size
    offsets = scene["isect_offsets"].flatten()
    ends = torch.cat([offsets[1:], offsets.new_tensor([scene["flatten_ids"].numel()])])
    assert (ends - offsets).max() > block_size

    alphas, last_ids = _forward(scene)
    assert alphas.max() > 0.999
    assert last_ids.max() >= block_size

    pixel_values = torch.rand((C, H, W, D), device=device)
    values, weights = _to_gaussians(scene, pixel_values, last_ids)
    ref_values = _vjp_reference(scene, pixel_values)
    torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-4)
    alpha_sums = alphas.sum((-3, -2, -1)).double()
    torch.testing.assert_close(
        weights.sum(-1).double(), alpha_sums, rtol=1e-4, atol=1e-3
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_invariants(tile_size, packed):
    C, N = 3, 80
    scene = _make_scene((2,), C, N, tile_size, packed, seed=3)
    I = scene["I"]
    alphas, last_ids = _forward(scene)

    ones = torch.ones((2, C, HEIGHT, WIDTH, 2), device=device)
    values, weights = _to_gaussians(scene, ones, last_ids)

    # A constant pixel value v accumulates to v * weight.
    torch.testing.assert_close(values[..., 0], weights, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(values[..., 1], weights, rtol=1e-5, atol=1e-6)

    # Per image, the accumulated weights sum to the rendered alpha sum.
    per_image = torch.zeros(I, dtype=torch.float64, device=device)
    per_image.index_add_(0, scene["image_ids"], weights.reshape(-1).double())
    alpha_sums = alphas.reshape(I, -1).double().sum(-1)
    torch.testing.assert_close(per_image, alpha_sums, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
@pytest.mark.parametrize("packed", [False, True])
def test_rasterization_meta(packed, rasterize_mode):
    torch.manual_seed(0)
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    B, C = 2, 3
    expand = lambda x: x.expand((B,) + x.shape)
    render_colors, render_alphas, meta = gsplat.rasterization(
        expand(means),
        expand(quats),
        expand(scales * 0.5),
        expand(opacities),
        expand(colors),
        expand(viewmats[:C]),
        expand(Ks[:C]),
        width,
        height,
        packed=packed,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
    )
    last_ids = meta["last_ids"]
    assert last_ids.dtype == torch.int32
    assert last_ids.shape == (B, C, height, width)

    scene = dict(
        means2d=meta["means2d"].detach().contiguous(),
        conics=meta["conics"].detach().contiguous(),
        opacities=meta["opacities"].detach().contiguous(),
        isect_offsets=meta["isect_offsets"],
        flatten_ids=meta["flatten_ids"],
        width=width,
        height=height,
        tile_size=meta["tile_size"],
        packed=packed,
    )
    # meta holds exactly what the forward consumed, including compensated
    # opacities. RGB+ED renders 4 channels; use the same kernel instantiation.
    alphas, fwd_last_ids = _forward(scene, channels=4)
    assert torch.equal(last_ids, fwd_last_ids)
    torch.testing.assert_close(alphas, render_alphas)

    gt = torch.rand_like(render_colors[..., :3])
    err = (render_colors[..., :3].detach() - gt).abs()
    values, weights = _to_gaussians(scene, err, last_ids)
    assert values.shape == meta["opacities"].shape + (3,)
    assert weights.shape == meta["opacities"].shape
    torch.testing.assert_close(values, _vjp_reference(scene, err), rtol=1e-4, atol=1e-3)

    if packed:
        image_ids = meta["batch_ids"] * C + meta["camera_ids"]
    else:
        n = weights.shape[-1]
        image_ids = torch.arange(B * C, device=device).repeat_interleave(n)
    per_image = torch.zeros(B * C, dtype=torch.float64, device=device)
    per_image.index_add_(0, image_ids, weights.reshape(-1).double())
    alpha_sums = render_alphas.reshape(B * C, -1).double().sum(-1)
    torch.testing.assert_close(per_image, alpha_sums, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("tile_size", [4, 16])
def test_edge_cases(tile_size):
    C, N = 2, 40
    scene = _make_scene((), C, N, tile_size, False, seed=4)
    _, last_ids = _forward(scene)
    rand = lambda D: torch.rand((C, HEIGHT, WIDTH, D), device=device)

    # D = 0 yields weights only, identical to any D > 0 run.
    values0, weights0 = _to_gaussians(scene, rand(0), last_ids)
    assert values0.shape == (C, N, 0)
    _, weights1 = _to_gaussians(scene, rand(1), last_ids)
    torch.testing.assert_close(weights0, weights1)

    # No intersections: zeros of the right shape.
    no_isects = dict(
        scene,
        flatten_ids=scene["flatten_ids"][:0],
        isect_offsets=torch.zeros_like(scene["isect_offsets"]),
    )
    values, weights = _to_gaussians(no_isects, rand(2), last_ids)
    assert values.shape == (C, N, 2) and not values.any() and not weights.any()

    with pytest.raises(ValueError):
        _to_gaussians(scene, rand(2).double(), last_ids)
    with pytest.raises(ValueError):
        _to_gaussians(scene, rand(2)[:, :-1], last_ids)
    with pytest.raises(ValueError):
        _to_gaussians(scene, rand(2), last_ids.long())


@pytest.mark.parametrize("D", [1, 3, 7])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_nan_pixel_reaches_only_its_contributors(tile_size, D):
    """A NaN pixel value makes NaN only the Gaussians blended into that pixel: a
    Gaussian that doesn't reach it is summed as if the value were 0 there, not
    poisoned by 0 * NaN (every lane of a warp takes part in each Gaussian's sum)."""
    C, N = 2, 60
    scene = _make_scene((), C, N, tile_size, False, seed=5)
    _, last_ids = _forward(scene)
    pixel_values = torch.rand((C, HEIGHT, WIDTH, D), device=device)
    pixel = (0, HEIGHT // 2, WIDTH // 2)
    zeroed = pixel_values.clone()
    zeroed[pixel] = 0.0
    one_hot = torch.zeros_like(pixel_values)
    one_hot[pixel] = 1.0
    contributors = _to_gaussians(scene, one_hot, last_ids)[0][..., 0] > 0  # [C, N]
    assert contributors.any() and not contributors.all()

    poisoned = pixel_values.clone()
    poisoned[pixel] = float("nan")
    values, _ = _to_gaussians(scene, poisoned, last_ids)
    expected, _ = _to_gaussians(scene, zeroed, last_ids)
    assert values[contributors].isnan().all()
    torch.testing.assert_close(values[~contributors], expected[~contributors])


@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_eval3d_has_no_last_ids():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    _, _, meta = gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats[:1],
        Ks[:1],
        width,
        height,
        packed=False,
        with_ut=True,
        with_eval3d=True,
    )
    assert meta["last_ids"] is None


# ---------------------------------------------------------------------------
# rasterize_to_gaussian_grids: the 3x3x3-grid version
# ---------------------------------------------------------------------------


def _random_gaussians(batch_dims, C, N, seed=5, flat=False):
    """3D Gaussians inside the view of C pinhole cameras, sized so that the closest points q of the
    pixel rays spread over about +-2 sigma (most of the 27 cells get used). The grid kernel takes its
    blending weights from the 2D scene and only q from these, so the two need not match."""
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*batch_dims, *s, device=device, generator=gen)
    means = torch.stack([(u(N) * 2 - 1) * 1.2, (u(N) * 2 - 1) * 0.9, u(N) + 2.5], -1)
    quats = torch.randn(*batch_dims, N, 4, device=device, generator=gen)
    scales = u(N, 3) * 0.7 + 0.3
    if flat:
        scales[
            ..., 2
        ] = 1e-6  # ReSplat's flat Gaussians: o and d are ~1e7 along the thin axis
    viewmats = torch.eye(4, device=device).repeat(*batch_dims, C, 1, 1)
    viewmats[..., :3, 3] = (u(C, 3) * 2 - 1) * 0.2
    Ks = torch.tensor(
        [[30.0, 0, WIDTH / 2], [0, 30.0, HEIGHT / 2], [0, 0, 1]], device=device
    )
    Ks = Ks.repeat(*batch_dims, C, 1, 1)
    return dict(means=means, quats=quats, scales=scales, viewmats=viewmats, Ks=Ks)


def _grid_reference(scene, pixel_values, gaussians, masks=None):
    """All contributors per pixel -> closest point q (rays from gaussian_ray_frames) -> trilinear -> index_add."""
    origins, dirs = gsplat.gaussian_ray_frames(
        **gaussians
    )  # [..., C, N, 3], [..., C, N, 3, 3]
    I, N, W, H = scene["I"], scene["N"], scene["width"], scene["height"]
    args = (
        scene["means2d"],
        scene["conics"],
        scene["opacities"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        W,
        H,
        scene["tile_size"],
    )
    ncg, _ = rasterize_num_contributing_gaussians(*args)
    ids, w = rasterize_contributing_gaussian_ids(*args, ncg)
    K, D = ids.shape[-1], pixel_values.shape[-1]
    ids, w = ids.reshape(I, H, W, K), w.reshape(I, H, W, K)
    if masks is not None:
        # masked tiles contribute nothing
        th, tw = scene["isect_offsets"].shape[-2:]
        m = masks.reshape(I, th, tw).repeat_interleave(scene["tile_size"], 1)
        m = m.repeat_interleave(scene["tile_size"], 2)[:, :H, :W]
        ids = torch.where(m[..., None], ids, -1)
    img, pi, pj, k = torch.where(ids >= 0)
    g = ids[img, pi, pj, k].long()
    rows = g if scene["packed"] else img * N + g
    o, A = origins.reshape(-1, 3)[rows], dirs.reshape(-1, 3, 3)[rows]
    p = torch.stack(
        [pj + 0.5, pi + 0.5, torch.ones_like(pi, dtype=torch.float)], -1
    ).float()
    d = (A @ p[..., None])[..., 0]
    q = torch.linalg.cross(d, torch.linalg.cross(o, d, dim=-1), dim=-1) / (d * d).sum(
        -1, keepdim=True
    )
    u = (q + 1).clamp(0, 2)
    i0 = (u >= 1).long()
    f = u - i0
    n_rows = origins.reshape(-1, 3).shape[0]
    values = torch.zeros(n_rows * 27, D, device=device)
    weights = torch.zeros(n_rows * 27, device=device)
    pv = pixel_values.reshape(I, H, W, D)[img, pi, pj]
    for c in range(8):
        bits = torch.tensor([c >> 2, (c >> 1) & 1, c & 1], device=device)
        tri = torch.where(bits.bool(), f, 1 - f).prod(-1)
        idx = i0 + bits
        cell = rows * 27 + idx[:, 0] * 9 + idx[:, 1] * 3 + idx[:, 2]
        wc = w[img, pi, pj, k] * tri
        weights.index_add_(0, cell, wc)
        values.index_add_(0, cell, wc[:, None] * pv)
    shape = scene["opacities"].shape
    return values.reshape(shape + (27, D)), weights.reshape(shape + (27,))


def _to_grids(scene, pixel_values, gaussians, last_ids, masks=None):
    return rasterize_to_gaussian_grids(
        scene["means2d"],
        scene["conics"],
        scene["opacities"],
        pixel_values,
        gaussians["means"],
        gaussians["quats"],
        gaussians["scales"],
        gaussians["viewmats"],
        gaussians["Ks"],
        scene["width"],
        scene["height"],
        scene["tile_size"],
        scene["isect_offsets"],
        scene["flatten_ids"],
        last_ids,
        masks=masks,
    )


@pytest.mark.parametrize("D", [1, 3, 4, 8])
@pytest.mark.parametrize("use_masks", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (2,)])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_grids_match_reference(tile_size, batch_dims, use_masks, D):
    C, N = 2, 60
    scene = _make_scene(batch_dims, C, N, tile_size, False, seed=6)
    masks = None
    if use_masks:
        gen = torch.Generator(device=device).manual_seed(8)
        masks = (
            torch.rand(scene["isect_offsets"].shape, device=device, generator=gen) > 0.3
        )
    _, last_ids = _forward(scene, masks)
    pixel_values = torch.randn(batch_dims + (C, HEIGHT, WIDTH, D), device=device)
    gaussians = _random_gaussians(batch_dims, C, N)

    values, weights = _to_grids(scene, pixel_values, gaussians, last_ids, masks)
    ref_values, ref_weights = _grid_reference(scene, pixel_values, gaussians, masks)
    assert values.shape == scene["opacities"].shape + (27, D)
    filled = (ref_weights > 0).sum(-1)
    assert (
        filled[filled > 0].float().mean() > 4
    )  # the cells are really spread (masks remove pixels)
    torch.testing.assert_close(values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-4, atol=1e-4)

    # Summed over cells: exactly rasterize_to_gaussians (trilinear weights sum to 1).
    flat_values, flat_weights = _to_gaussians(scene, pixel_values, last_ids, masks)
    torch.testing.assert_close(values.sum(-2), flat_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weights.sum(-1), flat_weights, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tile_size", [4, 16])
def test_grids_flat_gaussians(tile_size):
    """Flat Gaussians (one scale 1e-6, as ReSplat's): o and d are ~1e7 along the thin axis, and the kernel still
    matches the reference."""
    C, N, D = 2, 60, 4
    scene = _make_scene((), C, N, tile_size, False, seed=6)
    _, last_ids = _forward(scene)
    pixel_values = torch.randn((C, HEIGHT, WIDTH, D), device=device)
    gaussians = _random_gaussians((), C, N, flat=True)
    values, weights = _to_grids(scene, pixel_values, gaussians, last_ids)
    ref_values, ref_weights = _grid_reference(scene, pixel_values, gaussians)
    torch.testing.assert_close(values, ref_values, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-4, atol=1e-4)


def test_grids_saturation():
    # Contributors past the first batch and saturated pixels, as in the scalar test.
    C, N, D, W, H = 1, 4000, 4, 64, 64
    scene = _make_scene((), C, N, 16, False, W, H, seed=2, opacity_range=(0.05, 0.3))
    _, last_ids = _forward(scene)
    pixel_values = torch.rand((C, H, W, D), device=device)
    gaussians = _random_gaussians((), C, N)
    values, weights = _to_grids(scene, pixel_values, gaussians, last_ids)
    ref_values, ref_weights = _grid_reference(scene, pixel_values, gaussians)
    torch.testing.assert_close(values, ref_values, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-3, atol=1e-4)


def test_grids_bad_inputs():
    C, N = 2, 40
    scene = _make_scene((), C, N, 16, False, seed=4)
    _, last_ids = _forward(scene)
    gaussians = _random_gaussians((), C, N)
    with pytest.raises(ValueError):  # D = 9 is past the templated channel counts
        _to_grids(
            scene, torch.rand((C, HEIGHT, WIDTH, 9), device=device), gaussians, last_ids
        )
    with pytest.raises(ValueError):
        bad = dict(gaussians, means=gaussians["means"][:-1])
        _to_grids(
            scene, torch.rand((C, HEIGHT, WIDTH, 3), device=device), bad, last_ids
        )
    packed = _make_scene((), C, N, 16, True, seed=4)
    _, packed_last_ids = _forward(packed)
    with pytest.raises(ValueError):  # dense layout only
        _to_grids(
            packed,
            torch.rand((C, HEIGHT, WIDTH, 3), device=device),
            gaussians,
            packed_last_ids,
        )


@pytest.mark.parametrize("flat", [False, True])
def test_gaussian_ray_frames_closest_point(flat):
    """q from gaussian_ray_frames equals the Mahalanobis-closest point of the world ray, in the Gaussian's frame,
    also for flat Gaussians (one scale 1e-6, as ReSplat's), where o and d are ~1e7 along the thin axis."""
    from gsplat.cuda._math import _quat_to_rotmat

    gen = torch.Generator(device=device).manual_seed(0)
    N, C = 50, 3
    means = torch.randn(N, 3, device=device, generator=gen)
    quats = torch.randn(N, 4, device=device, generator=gen)
    scales = torch.rand(N, 3, device=device, generator=gen) * 0.3 + 0.05
    if flat:
        scales[:, 2] = 1e-6
    viewmats = torch.eye(4, device=device).repeat(C, 1, 1)
    viewmats[:, :3, 3] = torch.tensor(
        [[0.0, 0.0, 5.0], [0.5, -0.2, 6.0], [-0.4, 0.3, 4.0]], device=device
    )
    Ks = torch.tensor([[80.0, 0, 40], [0, 80.0, 30], [0, 0, 1]], device=device).repeat(
        C, 1, 1
    )
    origins, dirs = gsplat.gaussian_ray_frames(means, quats, scales, viewmats, Ks)
    assert origins.shape == (C, N, 3) and dirs.shape == (C, N, 3, 3)

    px = torch.tensor([23.5, 17.5, 1.0], device=device)
    d = dirs @ px
    q = torch.linalg.cross(d, torch.linalg.cross(origins, d, dim=-1), dim=-1) / (
        d * d
    ).sum(-1, keepdim=True)

    R, S = _quat_to_rotmat(quats).double(), scales.double()
    c2w = torch.linalg.inv(viewmats.double())
    k_inv_px = torch.linalg.solve(
        Ks.double(), px.double().expand(C, 3)[..., None]
    )  # [C, 3, 1]
    ray = (c2w[:, :3, :3] @ k_inv_px)[:, None, :, 0]  # [C, 1, 3]
    prec = R @ torch.diag_embed(S**-2) @ R.transpose(-1, -2)  # [N, 3, 3]
    v = c2w[:, None, :3, 3] - means.double()  # [C, N, 3]
    num = (v[..., None, :] @ prec @ ray[..., None])[..., 0, 0]
    den = (ray[..., None, :] @ prec @ ray[..., None])[..., 0, 0]
    x = v - (num / den)[..., None] * ray
    q_ref = ((R.transpose(-1, -2) @ x[..., None])[..., 0] / S).float()
    torch.testing.assert_close(q, q_ref, rtol=1e-4, atol=1e-3)
