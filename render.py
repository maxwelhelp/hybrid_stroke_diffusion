"""Adapter from flat clothoid parameters to VecGPT's differentiable renderer."""

from __future__ import annotations

import torch

try:
    from vecgpt.clothoid_render import render_clothoid_batch as _render_batch
except ImportError:
    from .vecgpt.clothoid_render import render_clothoid_batch as _render_batch
import inspect


def render_strokes(
    params: torch.Tensor,
    presence: torch.Tensor | None = None,
    *,
    size: int = 64,
    curve_samples: int = 32,
) -> torch.Tensor:
    """Render ``[B,N,10]`` clothoid strokes to ``[B,H,W,3]``."""
    if params.ndim != 3 or params.shape[-1] < 10:
        raise ValueError("params must have shape [B,N,10]")
    if presence is None:
        presence = torch.ones(params.shape[:2], device=params.device, dtype=params.dtype)

    sig = inspect.signature(_render_batch)
    # Adapt to whichever version is available
    if 'valid' in sig.parameters:
        rgba = torch.cat((params[..., 7:10], presence[..., None].clamp(0, 1)), -1)
        return _render_batch(
            params[..., :3], params[..., 4], params[..., 3], params[..., 5],
            params[..., 6], rgba, valid=presence > 0.01, size=size,
            curve_samples=curve_samples,
        )
    else:
        # Simplified stub: just pass params + presence
        return _render_batch(params, presence, size=size, curve_samples=curve_samples)
