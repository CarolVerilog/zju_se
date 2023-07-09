import itertools
from typing import Optional, Sequence, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from lpips import LPIPS
import tqdm

from ngp import NGPDensityField, NGPRadianceField
from datasets import DataLoader
from utils import (
    Rays,
    set_random_seed,
    namedtuple_map,
)

from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

from nerfacc.volrend import rendering

device = "cuda:0"
set_random_seed(42)


class NeRFacto:
    def __init__(self) -> None:
        # scene settings
        self.data_root = "data"
        self.train_split = "train"
        self.scene = "garden"
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        # train settings
        self.max_steps = 20000
        self.lr = 1e-2
        self.eps = 1e-15
        self.weight_decay = 0.0
        self.train_num_rays = 4096

        self.train_num_samples = 48
        self.train_num_samples_per_prop = [256, 9]
        self.train_near_plane = 0.2
        self.train_far_plane = 1e3

        self.render_num_samples = 48
        self.render_num_samples_per_prop = [256, 9]
        self.render_near_plane = 0.2
        self.render_far_plane = 1e3
        self.render_chunk_size = 8192

    def populate(self) -> None:
        # dataset parameters
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        # model parameters
        self.proposal_networks = [
            NGPDensityField(
                aabb=self.aabb,
                unbounded=True,
                n_levels=5,
                max_resolution=128,
            ).to(device),
            NGPDensityField(
                aabb=self.aabb,
                unbounded=True,
                n_levels=5,
                max_resolution=256,
            ).to(device),
        ]

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
        # setup the radiance field we want to train.
        self.prop_optimizer = torch.optim.Adam(
            itertools.chain(
                *[p.parameters() for p in self.proposal_networks],
            ),
            lr=self.lr,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.prop_optimizer,
                    milestones=[
                        self.max_steps // 2,
                        self.max_steps * 3 // 4,
                        self.max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        self.estimator = PropNetEstimator(self.prop_optimizer, self.prop_scheduler).to(
            device
        )

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
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

        lpips_net = LPIPS(net="vgg").to(device)
        lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        self.lpips_fn = lambda x, y: lpips_net(
            lpips_norm_fn(x), lpips_norm_fn(y)
        ).mean()

    def train(
        self,
        step,
    ):
        self.radiance_field.train()
        for p in self.proposal_networks:
            p.train()
        self.estimator.train()

        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        proposal_requires_grad = self.proposal_requires_grad_fn(step)
        # render
        rgb, _, _, extras = self.render(
            rays,
            # rendering options
            num_samples=self.train_num_samples,
            num_samples_per_prop=self.train_num_samples_per_prop,
            near_plane=self.train_near_plane,
            far_plane=self.train_far_plane,
            render_bkgd=render_bkgd,
            # train options
            proposal_requires_grad=proposal_requires_grad,
        )
        self.estimator.update_every_n_steps(
            extras["trans"], proposal_requires_grad, loss_scaler=1024
        )

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def eval(self, rays: Rays):
        self.radiance_field.eval()
        for p in self.proposal_networks:
            p.eval()
        self.estimator.eval()

        with torch.no_grad():
            rgb, _, _, _ = self.render(
                rays,
                num_samples=self.render_num_samples,
                num_samples_per_prop=self.render_num_samples_per_prop,
                near_plane=self.render_near_plane,
                far_plane=self.render_far_plane,
                render_bkgd=self.render_bkgd,
            )

        return rgb

    def test(self):
        # evaluation
        self.radiance_field.eval()
        for p in self.proposal_networks:
            p.eval()
        self.estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            print("testing nerf")
            for i in tqdm.tqdm(range(len(self.test_dataset))):
                data = self.test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                (
                    rgb,
                    _,
                    _,
                    _,
                ) = self.render(
                    rays,
                    # rendering options
                    num_samples=self.render_num_samples,
                    num_samples_per_prop=self.render_num_samples_per_prop,
                    near_plane=self.render_near_plane,
                    far_plane=self.render_far_plane,
                    render_bkgd=render_bkgd,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(self.lpips_fn(rgb, pixels).item())
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        return psnr_avg, lpips_avg

    def render(
        self,
        # scene
        rays: Rays,
        # rendering options
        num_samples: int,
        num_samples_per_prop: Sequence[int],
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        render_bkgd: Optional[torch.Tensor] = None,
        # train options
        proposal_requires_grad=False,
    ):
        """Render the pixels of an image."""
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins = chunk_rays.origins[..., None, :]
            t_dirs = chunk_rays.viewdirs[..., None, :]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            sigmas = proposal_network(positions)
            sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[..., None, :]
            t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            rgb, sigmas = self.radiance_field(positions, t_dirs)
            sigmas[..., -1, :] = torch.inf
            return rgb, sigmas.squeeze(-1)

        results = []
        chunk = (
            torch.iinfo(torch.int32).max
            if self.radiance_field.training
            else self.render_chunk_size
        )
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            t_starts, t_ends = self.estimator.sampling(
                prop_sigma_fns=[
                    lambda *args: prop_sigma_fn(*args, p)
                    for p in self.proposal_networks
                ],
                prop_samples=num_samples_per_prop,
                num_samples=num_samples,
                n_rays=chunk_rays.origins.shape[0],
                near_plane=near_plane,
                far_plane=far_plane,
                stratified=self.radiance_field.training,
                requires_grad=proposal_requires_grad,
            )
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices=None,
                n_rays=None,
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
            )
            chunk_results = [rgb, opacity, depth]
            results.append(chunk_results)

        colors, opacities, depths = collate(
            results,
            collate_fn_map={
                **default_collate_fn_map,
                torch.Tensor: lambda x, **_: torch.cat(x, 0),
            },
        )
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            extras,
        )
