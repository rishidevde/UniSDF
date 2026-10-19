"""
Proposal Sampling Network for UniSDF (Mip-NeRF 360 style).

Implements:
  - ProposalNetwork: lightweight hash-grid + MLP that predicts density
  - sample_along_rays: uniform initial sampling
  - resample_along_rays: PDF/CDF inverse-transform hierarchical resampling
  - proposal_loss: online distillation loss (bound final weights with proposal weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.network_utils import get_encoding, get_mlp


@models.register('proposal-network')
class ProposalNetwork(nn.Module):
    """
    A lightweight density-only network used for proposal sampling.
    Uses a small hash grid encoder + 2-layer MLP -> density (scalar).
    
    The same ProposalNetwork instance is called twice (for the two proposal rounds)
    to match the paper's parameter-sharing scheme.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_input_dims = 3
        self.encoding = get_encoding(self.n_input_dims, config.encoding_config)
        self.network = get_mlp(
            self.encoding.n_output_dims,
            1,  # output: density (scalar)
            config.mlp_network_config
        )
    
    def forward(self, positions):
        """
        Args:
            positions: (N, 3) contracted positions in [0, 1]
        Returns:
            density: (N,) non-negative density values
        """
        enc = self.encoding(positions.view(-1, 3))
        raw_density = self.network(enc).view(-1).float()
        # Softplus activation to ensure non-negative density
        density = F.softplus(raw_density - 1.0)  # bias of -1 for stability
        return density

    def update_step(self, epoch, global_step):
        """Forward the update to the encoding (for coarse-to-fine)."""
        if hasattr(self.encoding, 'update_step'):
            self.encoding.update_step(epoch, global_step)


# ============================================================
# Sampling utilities
# ============================================================

def sample_along_rays(rays_o, rays_d, near, far, n_samples, randomized=True):
    """
    Uniformly sample t-values along rays.
    
    Args:
        rays_o: (N_rays, 3)
        rays_d: (N_rays, 3)
        near: float or (N_rays, 1)
        far: float or (N_rays, 1)
        n_samples: int, number of samples
        randomized: bool, whether to add stratified noise
    
    Returns:
        t_vals: (N_rays, n_samples) sampled t-values
    """
    n_rays = rays_o.shape[0]
    device = rays_o.device
    
    # Uniform spacing in [0, 1]
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)  # (n_samples,)
    t_vals = t_vals.unsqueeze(0).expand(n_rays, -1)  # (N_rays, n_samples)
    
    if isinstance(near, (int, float)):
        near = torch.full((n_rays, 1), near, device=device)
    elif torch.is_tensor(near) and near.ndim == 1:
        near = near.unsqueeze(-1)
        
    if isinstance(far, (int, float)):
        far = torch.full((n_rays, 1), far, device=device)
    elif torch.is_tensor(far) and far.ndim == 1:
        far = far.unsqueeze(-1)
    
    # Map to [near, far]
    t_vals = near + (far - near) * t_vals  # (N_rays, n_samples)
    
    if randomized:
        # Stratified sampling: add uniform noise within each bin
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand
    
    return t_vals


def resample_along_rays(t_vals, weights, n_samples, randomized=True, eps=1e-5):
    """
    Resample t-values using inverse CDF of the weight distribution.
    This is the core of Mip-NeRF 360 proposal-based importance sampling.
    
    Args:
        t_vals: (N_rays, N_old) current t-values (bin edges)
        weights: (N_rays, N_old - 1) weights for each bin (from rendering or proposal)
        n_samples: int, number of new samples to draw
        randomized: bool, whether to add noise to CDF inversion
        eps: float, small value to prevent division by zero
    
    Returns:
        t_new: (N_rays, n_samples) resampled t-values, sorted
    """
    # Normalize weights to get PDF
    weights = weights + eps  # prevent all-zero
    pdf = weights / weights.sum(dim=-1, keepdim=True)  # (N_rays, N_bins)
    
    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)  # (N_rays, N_bins)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (N_rays, N_bins+1)
    
    # Uniform samples in [0, 1] for inverse CDF
    if randomized:
        u = torch.rand(cdf.shape[0], n_samples, device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - 1e-6, n_samples, device=cdf.device)
        u = u.unsqueeze(0).expand(cdf.shape[0], -1)
    u = u.contiguous()
    
    # Invert CDF via searchsorted
    # Find indices where u would be inserted into cdf
    indices = torch.searchsorted(cdf, u, right=True)  # (N_rays, n_samples)
    below = torch.clamp(indices - 1, min=0)
    above = torch.clamp(indices, max=cdf.shape[-1] - 1)
    
    # Gather CDF and t values at below/above
    cdf_below = torch.gather(cdf, dim=-1, index=below)
    cdf_above = torch.gather(cdf, dim=-1, index=above)
    
    # t_vals defines bin edges, so it has the same length as cdf
    t_below = torch.gather(t_vals, dim=-1, index=below)
    t_above = torch.gather(t_vals, dim=-1, index=above)
    
    # Linear interpolation
    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t_new = t_below + (u - cdf_below) / denom * (t_above - t_below)
    
    # Sort and clamp
    t_new, _ = torch.sort(t_new, dim=-1)
    
    return t_new


def compute_proposal_loss(prop_weights_list, final_weights, t_vals_list, final_t_vals):
    """
    Proposal loss from Mip-NeRF 360: ensures proposal weights upper-bound final weights.
    
    For each proposal level, we compute:
        L_prop = mean( max(0, w_final_hist - w_prop_hist)^2 )
    
    where w_final_hist and w_prop_hist are histograms of the final and proposal weights
    binned onto the proposal's t-value grid.
    
    Args:
        prop_weights_list: list of (N_rays, N_prop_bins) proposal weights per level
        final_weights: (N_rays, N_final_bins) final rendering weights  
        t_vals_list: list of (N_rays, N_prop_bins+1) proposal t-values per level
        final_t_vals: (N_rays, N_final_bins+1) final t-values
    
    Returns:
        loss: scalar proposal loss
    """
    loss = 0.0
    
    for prop_weights, prop_t_vals in zip(prop_weights_list, t_vals_list):
        # Resample the final weights onto the proposal grid
        # Compute histogram of final weights in proposal bins
        final_hist = _resample_weights_to_bins(
            final_t_vals, final_weights, prop_t_vals
        )
        
        # Bound loss: penalize when proposal underestimates final
        # Use L2 for smooth gradients
        diff = torch.clamp(final_hist - prop_weights, min=0.0)
        loss = loss + (diff ** 2).mean()
    
    return loss / max(len(prop_weights_list), 1)


def _resample_weights_to_bins(src_t_vals, src_weights, dst_t_vals):
    """
    Redistribute src_weights (defined on src_t_vals bins) into dst_t_vals bins.
    
    Args:
        src_t_vals: (N_rays, N_src+1) source bin edges
        src_weights: (N_rays, N_src) source bin weights
        dst_t_vals: (N_rays, N_dst+1) destination bin edges
    
    Returns:
        dst_weights: (N_rays, N_dst) resampled weights
    """
    n_rays = src_t_vals.shape[0]
    n_dst = dst_t_vals.shape[-1] - 1
    device = src_t_vals.device
    
    dst_weights = torch.zeros(n_rays, n_dst, device=device)
    
    # For each destination bin, compute overlap with source bins
    for j in range(n_dst):
        dst_lo = dst_t_vals[:, j:j+1]       # (N_rays, 1)
        dst_hi = dst_t_vals[:, j+1:j+2]     # (N_rays, 1)
        
        src_lo = src_t_vals[:, :-1]          # (N_rays, N_src)
        src_hi = src_t_vals[:, 1:]           # (N_rays, N_src)
        
        # Overlap between [dst_lo, dst_hi] and [src_lo, src_hi]
        overlap_lo = torch.max(dst_lo, src_lo)
        overlap_hi = torch.min(dst_hi, src_hi)
        overlap = torch.clamp(overlap_hi - overlap_lo, min=0.0)
        
        # Fraction of each source bin that falls in this dst bin
        src_width = (src_hi - src_lo).clamp(min=1e-8)
        frac = overlap / src_width
        
        dst_weights[:, j] = (frac * src_weights).sum(dim=-1)
    
    return dst_weights
