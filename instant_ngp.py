import itertools
from typing import Optional, Sequence, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tqdm

from ngp import NGPRadianceField
from datasets import DataLoader
from utils import (
    Rays,
    set_random_seed,
    namedtuple_map,
)

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    rendering,
    accumulate_along_rays_,
    render_weight_from_density,
)

device = "cuda:0"
set_random_seed(42)


class InstantNGP:
    def __init__(self) -> None:
        # scene settings
        self.data_root = "data"
        self.train_split = "train"
        self.scene = "bicycle"
        self.factor = 4
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

        # model settings
        self.grid_resolution = 128
        self.num_grid_levels = 4

        # train settings
        self.max_steps = 20000
        self.lr = 1e-2
        self.eps = 1e-15
        self.weight_decay = 0.0
        self.train_num_rays = 1024
        self.train_target_sample_batch_size = 1 << 18

        self.train_num_samples = 48
        self.train_near_plane = 0.2
        self.train_far_plane = 1e10
        self.train_render_step_size = 1e-3
        self.train_alpha_thre = 1e-2
        self.train_cone_angle = 0.004

        # test settings
        self.test_num_samples = 48
        self.test_near_plane = 0.2
        self.test_far_plane = 1e10
        self.test_chunk_size = 8192
        self.test_render_step_size = 1e-3
        self.test_alpha_thre = 1e-2
        self.test_cone_angle = 0.004

        # draw settings
        self.draw_num_samples = 48
        self.draw_near_plane = 0.2
        self.draw_far_plane = 1e10
        self.draw_chunk_size = 8192
        self.draw_render_step_size = 1e-3
        self.draw_alpha_thre = 1e-2
        self.draw_cone_angle = 0.004

        # metrics
        self.ssim = structural_similarity_index_measure
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    def populate(self) -> None:
        # dataset parameters
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": self.factor}
        test_dataset_kwargs = {"factor": self.factor}

        self.train_dataset = DataLoader(
            subject_id=self.scene,
            root_fp=self.data_root,
            split=self.train_split,
            num_rays=self.train_num_rays,
            device=device,
            **train_dataset_kwargs,
        )

        self.test_dataset = DataLoader(
            subject_id=self.scene,
            root_fp=self.data_root,
            split="test",
            num_rays=None,
            device=device,
            **test_dataset_kwargs,
        )

        self.render_bkgd = self.test_dataset[0]["color_bkgd"]

        self.estimator = OccGridEstimator(
            roi_aabb=self.aabb,
            resolution=self.grid_resolution,
            levels=self.num_grid_levels,
        ).to(device)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        self.radiance_field = NGPRadianceField(aabb=self.aabb, unbounded=True).to(
            device
        )
        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        self.max_steps // 2,
                        self.max_steps * 3 // 4,
                        self.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )

    def train(
        self,
        step,
    ):
        self.radiance_field.train()
        self.estimator.train()

        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.train_render_step_size

        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, _, _, extras = self.render(
            rays,
            # rendering options
            near_plane=self.train_near_plane,
            far_plane=self.train_far_plane,
            render_bkgd=render_bkgd,
            render_step_size=self.train_render_step_size,
            cone_angle=self.train_cone_angle,
            alpha_thre=self.train_alpha_thre,
        )

        if extras["num_samples"] == 0:
            return None

        if self.train_target_sample_batch_size > 0:
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (self.train_target_sample_batch_size / float(extras["num_samples"]))
            )
            self.train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def eval(self, rays: Rays):
        self.radiance_field.eval()
        self.estimator.eval()

        rgb, _, _, _ = self.render_image(
            rays,
            near_plane=self.test_near_plane,
            far_plane=self.test_far_plane,
            render_bkgd=self.render_bkgd,
            render_step_size=self.test_render_step_size,
            cone_angle=self.test_cone_angle,
            alpha_thre=self.test_alpha_thre,
        )

        return rgb

    @torch.no_grad()
    def draw(self, rays: Rays):
        self.radiance_field.eval()
        self.estimator.eval()

        rgb, _, _, _ = self.render_image(
            rays,
            near_plane=self.draw_near_plane,
            far_plane=self.draw_far_plane,
            render_bkgd=self.render_bkgd,
            render_step_size=self.draw_render_step_size,
            cone_angle=self.draw_cone_angle,
            alpha_thre=self.draw_alpha_thre,
        )

        return rgb

    @torch.no_grad()
    def test(self):
        # evaluation
        self.radiance_field.eval()
        self.estimator.eval()

        ssims = []
        psnrs = []
        lpips = []

        print("Testing nerf")
        for i in tqdm.tqdm(range(len(self.test_dataset))):
            data = self.test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"].to(device)

            # rendering
            (
                rgb,
                _,
                _,
                _,
            ) = self.render_image(
                rays,
                # rendering options
                near_plane=self.test_near_plane,
                far_plane=self.test_far_plane,
                render_bkgd=render_bkgd,
                render_step_size=self.test_render_step_size,
                cone_angle=self.test_cone_angle,
                alpha_thre=self.test_alpha_thre,
            )

            rgb = torch.permute(rgb, (2, 0, 1)).unsqueeze(0)
            pixels = torch.permute(pixels, (2, 0, 1)).unsqueeze(0)
            ssims.append(self.ssim(rgb, pixels))
            psnrs.append(self.psnr(rgb, pixels))
            lpips.append(self.lpips(rgb, pixels))

        ssim = (sum(ssims) / len(ssims)).item()
        psnr = (sum(psnrs) / len(psnrs)).item()
        lpips = (sum(lpips) / len(lpips)).item()

        return ssim, psnr, lpips

    def render(
        self,
        # scene
        rays: Rays,
        # rendering options
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        render_bkgd: Optional[torch.Tensor] = None,
        chunk_size: int = None,
        render_step_size: int = 1e-3,
        cone_angle: float = 0.004,
        alpha_thre: float = 1e-2,
    ):
        """Render the pixels of an image."""
        rays_shape = rays.origins.shape
        num_rays, _ = rays_shape

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            sigmas = self.radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            rgb, sigmas = self.radiance_field(positions, t_dirs)
            return rgb, sigmas.squeeze(-1)

        results = []
        num_samples = 0

        chunk = torch.iinfo(torch.int32).max if chunk_size == None else chunk_size
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o=rays.origins,
                rays_d=rays.viewdirs,
                sigma_fn=sigma_fn,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices=ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [rgb, opacity, depth]
            results.append(chunk_results)
            num_samples += len(t_starts)

        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )

        extras["num_samples"] = num_samples

        return (
            torch.clamp(colors, min=0.0, max=1.0).view((*rays_shape[:-1], -1)),
            torch.clamp(opacities, min=0.0, max=1.0).view((*rays_shape[:-1], -1)),
            torch.clamp(depths, min=0.0, max=1.0).view((*rays_shape[:-1], -1)),
            extras,
        )

    @torch.no_grad()
    def render_image(
        self,
        rays: Rays,
        max_samples: int = 1024,
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        render_step_size: float = 1e-3,
        render_bkgd: Optional[torch.Tensor] = None,
        cone_angle: float = 0.0,
        alpha_thre: float = 0.0,
        early_stop_eps: float = 1e-4,
    ):
        rays_shape = rays.origins.shape
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays.origins[ray_indices]
            t_dirs = rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0

            rgbs, sigmas = self.radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        device = rays.origins.device
        opacity = torch.zeros(num_rays, 1, device=device)
        depth = torch.zeros(num_rays, 1, device=device)
        rgb = torch.zeros(num_rays, 3, device=device)

        ray_mask = torch.ones(num_rays, device=device).bool()

        min_samples = 4

        iter_samples = total_samples = 0

        rays_o = rays.origins
        rays_d = rays.viewdirs

        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, self.estimator.aabbs)
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)

        opc_thre = 1 - early_stop_eps

        while iter_samples < max_samples:
            n_alive = ray_mask.sum().item()
            if n_alive == 0:
                break

            # the number of samples to add on each ray
            n_samples = max(min(num_rays // n_alive, 64), min_samples)
            iter_samples += n_samples

            (intervals, samples, termination_planes) = traverse_grids(
                rays_o,
                rays_d,
                self.estimator.binaries,
                self.estimator.aabbs,
                near_planes,
                far_planes,
                render_step_size,
                cone_angle,
                n_samples,
                True,
                ray_mask,
                t_sorted,
                t_indices,
                hits,
            )
            t_starts = intervals.vals[intervals.is_left]
            t_ends = intervals.vals[intervals.is_right]
            ray_indices = samples.ray_indices[samples.is_valid]
            packed_info = samples.packed_info

            rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
            weights, _, alphas = render_weight_from_density(
                t_starts,
                t_ends,
                sigmas,
                ray_indices=ray_indices,
                n_rays=num_rays,
                prefix_trans=1 - opacity[ray_indices].squeeze(-1),
            )
            if alpha_thre > 0:
                vis_mask = alphas >= alpha_thre
                ray_indices, rgbs, weights, t_starts, t_ends = (
                    ray_indices[vis_mask],
                    rgbs[vis_mask],
                    weights[vis_mask],
                    t_starts[vis_mask],
                    t_ends[vis_mask],
                )

            accumulate_along_rays_(
                weights,
                values=rgbs,
                ray_indices=ray_indices,
                outputs=rgb,
            )
            accumulate_along_rays_(
                weights,
                values=None,
                ray_indices=ray_indices,
                outputs=opacity,
            )
            accumulate_along_rays_(
                weights,
                values=(t_starts + t_ends)[..., None] / 2.0,
                ray_indices=ray_indices,
                outputs=depth,
            )

            near_planes = termination_planes
            ray_mask = torch.logical_and(
                opacity.view(-1) <= opc_thre,
                packed_info[:, 1] == n_samples,
            )
            total_samples += ray_indices.shape[0]

        rgb = rgb + render_bkgd * (1.0 - opacity)
        depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

        extras = {}
        extras["num_samples"] = total_samples

        return (
            torch.clamp(rgb, min=0.0,max=1.0).view((*rays_shape[:-1], -1)),
            torch.clamp(opacity, min=0.0, max=1.0).view((*rays_shape[:-1], -1)),
            torch.clamp(depth, min=0.0, max=1.0).view((*rays_shape[:-1], -1)),
            extras,
        )
