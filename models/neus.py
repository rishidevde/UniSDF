"""
UniSDF NeuS Model with:
  - VolSDF Laplace density conversion (replaces NeuS logistic)
  - Mip-NeRF 360 proposal sampling (replaces nerfacc occupancy grid for foreground)
  - Dual radiance field composition (camera + reflection)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch, scale_anything
from systems.utils import update_module_step
from models.proposal import (
    ProposalNetwork,
    sample_along_rays,
    resample_along_rays,
    compute_proposal_loss,
)

# nerfacc is still used for the background model only
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect


# ============================================================
# Laplace Density Helpers (VolSDF formulation)
# ============================================================

def laplace_cdf(sdf, beta):
    """
    Compute the Laplace CDF: Ψ_β(s).
    (Kept for reference only, not used in the active density pipeline)
    
    Ψ_β(s) = 0.5 * exp(s/β)           if s <= 0
    Ψ_β(s) = 1 - 0.5 * exp(-s/β)      if s > 0
    
    Args:
        sdf: (...) signed distance values
        beta: scalar, positive scale parameter
    
    Returns:
        cdf: (...) values in [0, 1]
    """
    # Numerically stable implementation
    return torch.where(
        sdf <= 0,
        0.5 * torch.exp(sdf / beta),
        1.0 - 0.5 * torch.exp(-sdf / beta)
    )


def laplace_density(sdf, beta, alpha_val=1.0):
    """
    Convert SDF to density using the Laplace distribution (VolSDF, Eq. 4 in UniSDF).
    
    σ(s) = α * Ψ'_β(s) = α / (2β) * exp(-|s|/β)
    
    Args:
        sdf: (...) signed distance values
        beta: scalar, positive scale parameter
        alpha_val: scalar, density scale factor
    
    Returns:
        density: (...) non-negative density values
    """
    return alpha_val / (2.0 * beta) * torch.exp(-sdf.abs() / beta)


def sdf_to_alpha(sdf, beta, dists):
    """
    Convert SDF values to alpha (opacity) using the Laplace CDF formulation.
    This is the VolSDF approach used in UniSDF.
    
    α_i = max(Ψ_β(d_i) - Ψ_β(d_{i+1}), 0) / Ψ_β(d_i)
    
    which simplifies (via the approximation used in VolSDF) to:
    α_i = 1 - exp(-σ_i * Δt_i)
    
    Args:
        sdf: (N,) signed distance at sample points
        beta: scalar, positive Laplace scale parameter
        dists: (N,) distance between adjacent samples
    
    Returns:
        alpha: (N,) opacity values in [0, 1]
    """
    density = laplace_density(sdf, beta)
    alpha = 1.0 - torch.exp(-density * dists)
    return alpha.clamp(0.0, 1.0)


# ============================================================
# Volumetric Rendering Helpers (no nerfacc dependency)
# ============================================================

def compute_transmittance_weights(alpha):
    """
    Compute transmittance and rendering weights from alpha values.
    
    T_i = prod_{j=1}^{i-1} (1 - α_j)
    w_i = T_i * α_i
    
    Args:
        alpha: (N_rays, N_samples) opacity values
    
    Returns:
        weights: (N_rays, N_samples) rendering weights
        transmittance: (N_rays, N_samples) accumulated transmittance
    """
    # Exclusive cumprod: T_i = prod_{j<i} (1 - alpha_j)
    # Pad with 1.0 at the beginning and remove last element
    one_minus_alpha = 1.0 - alpha + 1e-10  # numerical stability
    # cumprod along samples dimension, shifted by 1
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), one_minus_alpha[:, :-1]], dim=-1),
        dim=-1
    )
    weights = alpha * transmittance
    return weights, transmittance


def accumulate_along_rays_dense(weights, values=None):
    """
    Accumulate values along rays using per-ray weights.
    Dense version (no ray_indices needed since all rays have same sample count).
    
    Args:
        weights: (N_rays, N_samples)
        values: (N_rays, N_samples) or (N_rays, N_samples, C), optional
    
    Returns:
        accumulated: (N_rays,) if values is None (opacity),
                     (N_rays, C) if values has channels
    """
    if values is None:
        return weights.sum(dim=-1, keepdim=True)  # opacity
    
    if values.ndim == 2:
        # (N_rays, N_samples) -> (N_rays,)
        return (weights * values).sum(dim=-1, keepdim=True)
    else:
        # (N_rays, N_samples, C) -> (N_rays, C)
        return (weights.unsqueeze(-1) * values).sum(dim=-2)


# ============================================================
# NeuS Model with UniSDF modifications
# ============================================================

@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.UN_BOUNDED_SPHERE

        # --- Background model (still uses nerfacc) ---
        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01

        # --- Learnable beta for Laplace density (VolSDF) ---
        # We store log_beta and use softplus to guarantee positivity
        beta_init = self.config.get('beta_init', 0.1)
        self.register_parameter(
            'log_beta',
            nn.Parameter(torch.tensor(math.log(max(beta_init, 1e-4))))
        )

        # --- Proposal Network (Mip-NeRF 360 style) ---
        proposal_config = self.config.get('proposal', None)
        if proposal_config is not None:
            self.proposal_network = ProposalNetwork(proposal_config)
            self.n_prop_samples_1 = proposal_config.get('n_samples_1', 64)
            self.n_prop_samples_2 = proposal_config.get('n_samples_2', 32)
            self.n_final_samples = proposal_config.get('n_final_samples', 32)
            self.use_proposal = True
        else:
            # Fallback: use nerfacc occupancy grid if no proposal config
            self.use_proposal = False
            self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        # --- Scene AABB and occupancy grid ---
        self.register_buffer('scene_aabb', torch.as_tensor(
            [-self.config.radius, -self.config.radius, -self.config.radius,
              self.config.radius,  self.config.radius,  self.config.radius],
            dtype=torch.float32
        ))

        if self.config.grid_prune:
            if not self.use_proposal:
                # Only create foreground occ grid if NOT using proposal sampling
                self.occupancy_grid = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=128,
                    contraction_type=ContractionType.AABB
                )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )

        self.randomized = self.config.randomized
        self.background_color = None

    @property
    def beta(self):
        """Positive beta parameter for Laplace density."""
        return F.softplus(self.log_beta)

    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        if self.use_proposal:
            self.proposal_network.update_step(epoch, global_step)

        if not self.training:
            return

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        # Background occupancy grid update (nerfacc)
        if self.config.learned_background and self.config.grid_prune:
            def occ_eval_fn_bg(x):
                density, _ = self.geometry_bg(x)
                return density[..., None] * self.render_step_size_bg

            self.occupancy_grid_bg.every_n_step(
                step=global_step, occ_eval_fn=occ_eval_fn_bg,
                occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01)
            )

        # Foreground occupancy grid update (only if NOT using proposal)
        if not self.use_proposal and self.config.grid_prune:
            def occ_eval_fn(x):
                sdf = self.geometry(x, with_grad=False, with_feature=False)
                density = laplace_density(sdf, self.beta)
                return density[..., None] * self.render_step_size
            self.occupancy_grid.every_n_step(
                step=global_step, occ_eval_fn=occ_eval_fn,
                occ_thre=self.config.get('grid_prune_occ_thre', 0.01)
            )

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, dists):
        """
        Convert SDF to alpha using VolSDF Laplace formulation.
        Replaces the old NeuS logistic density.
        
        Args:
            sdf: (N,) signed distance values
            dists: (N,) interval widths
        
        Returns:
            alpha: (N,) opacity in [0, 1]
        """
        return sdf_to_alpha(sdf, self.beta, dists)

    # ============================================================
    # Background rendering (unchanged, uses nerfacc)
    # ============================================================
    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            return density[..., None]

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions)
        rgb = self.texture_bg(feature, t_dirs)

        weights = render_weight_from_density(t_starts, t_ends, density[..., None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights_bg': weights.view(-1),
                'points_bg': midpoints.view(-1),
                'intervals_bg': intervals.view(-1),
                'ray_indices_bg': ray_indices.view(-1)
            })

        return out

    # ============================================================
    # Foreground rendering with Proposal Sampling + Laplace Density
    # ============================================================
    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # (N_rays, 3)

        if self.use_proposal:
            return self._forward_proposal(rays_o, rays_d, n_rays, rays)
        else:
            return self._forward_occupancy_grid(rays_o, rays_d, n_rays, rays)

    def _forward_proposal(self, rays_o, rays_d, n_rays, rays):
        """
        Forward pass using Mip-NeRF 360 proposal sampling + Laplace density.
        3-stage hierarchical sampling: 64 → 32 → 32.
        """
        device = rays_o.device
        radius = self.config.radius

        # Near/far from AABB intersection
        t_min, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # Handle rays that miss the AABB (t_max > 1e9) and clamp to valid ranges
        near = t_min.clamp(min=0.001)
        far = t_max.clamp(min=0.001, max=1e3)
        far = torch.maximum(near + 1e-3, far)

        # --- Stage 1: Uniform sampling (64 samples) ---
        t_vals_1 = sample_along_rays(rays_o, rays_d, near, far, self.n_prop_samples_1 + 1, self.randomized)
        # t_vals_1: (N_rays, 65) bin edges
        midpoints_1 = 0.5 * (t_vals_1[:, :-1] + t_vals_1[:, 1:])  # (N_rays, 64) bin centers
        positions_1 = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * midpoints_1.unsqueeze(-1)  # (N_rays, 64, 3)

        # Contract positions and evaluate proposal network
        positions_1_contracted = self._contract_positions(positions_1.reshape(-1, 3))
        prop_density_1 = self.proposal_network(positions_1_contracted).reshape(n_rays, -1)  # (N_rays, 64)

        # Compute proposal weights
        dists_1 = t_vals_1[:, 1:] - t_vals_1[:, :-1]  # (N_rays, 64)
        prop_alpha_1 = 1.0 - torch.exp(-prop_density_1 * dists_1)
        prop_weights_1, _ = compute_transmittance_weights(prop_alpha_1)  # (N_rays, 64)

        # --- Stage 2: Resample 32 points from proposal 1 ---
        t_vals_2 = resample_along_rays(t_vals_1, prop_weights_1, self.n_prop_samples_2 + 1, self.randomized)
        midpoints_2 = 0.5 * (t_vals_2[:, :-1] + t_vals_2[:, 1:])  # (N_rays, 32)
        positions_2 = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * midpoints_2.unsqueeze(-1)

        positions_2_contracted = self._contract_positions(positions_2.reshape(-1, 3))
        prop_density_2 = self.proposal_network(positions_2_contracted).reshape(n_rays, -1)

        dists_2 = t_vals_2[:, 1:] - t_vals_2[:, :-1]
        prop_alpha_2 = 1.0 - torch.exp(-prop_density_2 * dists_2)
        prop_weights_2, _ = compute_transmittance_weights(prop_alpha_2)

        # --- Stage 3: Resample 32 final points from proposal 2 ---
        t_vals_final = resample_along_rays(t_vals_2, prop_weights_2, self.n_final_samples + 1, self.randomized)
        midpoints_final = 0.5 * (t_vals_final[:, :-1] + t_vals_final[:, 1:])  # (N_rays, 32)
        dists_final = t_vals_final[:, 1:] - t_vals_final[:, :-1]  # (N_rays, 32)

        # --- Evaluate main SDF network at final sample positions ---
        positions_final = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * midpoints_final.unsqueeze(-1)  # (N_rays, 32, 3)
        positions_flat = positions_final.reshape(-1, 3)
        dirs_flat = rays_d.unsqueeze(1).expand_as(positions_final).reshape(-1, 3)

        predict_normal = getattr(self.geometry, 'predict_normal', False)
        if predict_normal:
            sdf, sdf_grad, feature, sdf_laplace, pred_normal = self.geometry(
                positions_flat, with_grad=True, with_feature=True, with_laplace=True, with_predicted_normal=True
            )
        else:
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(
                positions_flat, with_grad=True, with_feature=True, with_laplace=True
            )
            pred_normal = None

        normal = F.normalize(sdf_grad, p=2, dim=-1)

        # --- Laplace density → alpha ---
        alpha = self.get_alpha(sdf, dists_final.reshape(-1))  # (N_rays*32,)
        alpha = alpha.reshape(n_rays, self.n_final_samples)  # (N_rays, 32)

        # --- Compute rendering weights ---
        weights, transmittance = compute_transmittance_weights(alpha)  # (N_rays, 32)

        # --- Evaluate radiance ---
        if self.config.texture.name == 'unisdf-radiance':
            cam_color, ref_color, ref_weight = self.texture(feature, dirs_flat, normal)
            # Reshape to (N_rays, N_samples, C)
            cam_color = cam_color.reshape(n_rays, self.n_final_samples, 3)
            ref_color = ref_color.reshape(n_rays, self.n_final_samples, 3)
            ref_weight = ref_weight.reshape(n_rays, self.n_final_samples, 1)

            # Accumulate each radiance independently, then compose (Eq. 8)
            comp_cam_rgb = accumulate_along_rays_dense(weights, cam_color)  # (N_rays, 3)
            comp_ref_rgb = accumulate_along_rays_dense(weights, ref_color)  # (N_rays, 3)
            comp_weight = accumulate_along_rays_dense(weights, ref_weight.squeeze(-1))  # (N_rays, 1)

            # Pixel-level composition
            comp_rgb = comp_weight * comp_ref_rgb + (1.0 - comp_weight) * comp_cam_rgb
        else:
            rgb = self.texture(feature, dirs_flat, normal)
            rgb = rgb.reshape(n_rays, self.n_final_samples, 3)
            comp_rgb = accumulate_along_rays_dense(weights, rgb)

        opacity = weights.sum(dim=-1, keepdim=True)  # (N_rays, 1)
        depth = accumulate_along_rays_dense(weights, midpoints_final)  # (N_rays, 1)

        # Normal compositing
        normal_reshaped = normal.reshape(n_rays, self.n_final_samples, 3)
        comp_normal = accumulate_along_rays_dense(weights, normal_reshaped)  # (N_rays, 3)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        rays_fg = opacity > 0.1
        comp_normal[~rays_fg.expand_as(comp_normal)] = 0

        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([n_rays * self.n_final_samples], dtype=torch.int32, device=device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.reshape(-1),
                'points': midpoints_final.reshape(-1),
                'intervals': dists_final.reshape(-1),
                'ray_indices': torch.arange(n_rays, device=device).unsqueeze(1).expand(-1, self.n_final_samples).reshape(-1),
                'sdf_laplace_samples': sdf_laplace,
                't_dirs': dirs_flat,
                # Proposal data for proposal loss
                'prop_weights_list': [prop_weights_1, prop_weights_2],
                'final_weights': weights,
                'prop_t_vals_list': [t_vals_1, t_vals_2],
                'final_t_vals': t_vals_final,
            })
            if pred_normal is not None:
                out['pred_normal_samples'] = pred_normal

        # --- Background ---
        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None, :].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def _contract_positions(self, positions):
        """Contract positions using unbounded sphere contraction (matches geometry.py)."""
        from models.geometry import contract_to_unisphere
        return contract_to_unisphere(positions, self.config.radius, ContractionType.UN_BOUNDED_SPHERE)

    def _forward_occupancy_grid(self, rays_o, rays_d, n_rays, rays):
        """
        Fallback forward pass using nerfacc occupancy grid.
        Kept for backward compatibility when proposal config is not set.
        """
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        predict_normal = getattr(self.geometry, 'predict_normal', False)
        if predict_normal:
            sdf, sdf_grad, feature, sdf_laplace, pred_normal = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True, with_predicted_normal=True)
        else:
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
            pred_normal = None
        normal = F.normalize(sdf_grad, p=2, dim=-1)

        # Use Laplace density (VolSDF) instead of NeuS logistic
        alpha = self.get_alpha(sdf, dists.reshape(-1))[..., None]

        if self.config.texture.name == 'unisdf-radiance':
            cam_color, ref_color, ref_weight = self.texture(feature, t_dirs, normal)
            weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
            opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
            depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)

            comp_cam_rgb = accumulate_along_rays(weights, ray_indices, values=cam_color, n_rays=n_rays)
            comp_ref_rgb = accumulate_along_rays(weights, ray_indices, values=ref_color, n_rays=n_rays)
            comp_weight = accumulate_along_rays(weights, ray_indices, values=ref_weight, n_rays=n_rays)

            comp_rgb = comp_weight * comp_ref_rgb + (1 - comp_weight) * comp_cam_rgb
        else:
            rgb = self.texture(feature, t_dirs, normal)
            weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
            opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
            depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
            comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        rays_fg = opacity > 0.1
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        comp_normal[~rays_fg[:, 0]] = 0

        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1),
                'sdf_laplace_samples': sdf_laplace,
            })
            if pred_normal is not None:
                out['pred_normal_samples'] = pred_normal
            out['t_dirs'] = t_dirs

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None, :].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    def forward(self, rays):
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {
            **out,
            'beta': self.beta  # log beta for monitoring (replaces inv_s)
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()

    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, features = chunk_batch(self.geometry, export_config.chunk_size, True, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            base_color = torch.sigmoid(features[..., 1:4])
            mesh['v_rgb'] = base_color.cpu()
            mesh['v_norm'] = normal.cpu()
        return mesh