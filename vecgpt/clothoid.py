"""Clothoid curve utilities."""
import torch
import math

def clothoid_points(anchor, kappa, length, delta_kappa, samples=24):
    """Sample points along a clothoid curve.
    
    Args:
        anchor: [..., 3] tensor of (x, y, theta)
        kappa: [...] curvature
        length: [...] arc length
        delta_kappa: [...] curvature derivative
        samples: number of sample points
    
    Returns:
        [..., samples, 2] tensor of 2D points
    """
    # Get batch dimensions
    batch_shape = anchor.shape[:-1]
    
    # Sample along the curve
    t = torch.linspace(0, 1, samples, device=anchor.device, dtype=anchor.dtype)
    s = t * length[..., None]  # [..., samples]
    
    # Clothoid parametric equations (simplified)
    # For a true clothoid, integrate the tangent angle
    theta_0 = anchor[..., 2:3]  # [..., 1]
    theta = theta_0 + kappa[..., None] * s + 0.5 * delta_kappa[..., None] * s.square()
    
    # Integrate to get positions
    dx = torch.cos(theta)
    dy = torch.sin(theta)
    
    # Cumulative sum for integration
    x = torch.cumsum(dx, dim=-1) * (length[..., None] / samples)
    y = torch.cumsum(dy, dim=-1) * (length[..., None] / samples)
    
    # Rotate and translate to anchor position
    cos_t0 = torch.cos(theta_0)
    sin_t0 = torch.sin(theta_0)
    x_rot = x * cos_t0 - y * sin_t0
    y_rot = x * sin_t0 + y * cos_t0
    
    points = torch.stack([
        anchor[..., 0:1] + x_rot,
        anchor[..., 1:2] + y_rot
    ], dim=-1)  # [..., samples, 2]
    
    return points
