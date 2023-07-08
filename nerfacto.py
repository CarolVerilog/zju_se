import itertools
from typing import Optional, Sequence, Literal

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from lpips import LPIPS

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
    def __init__(
        self,
        data_root,
        train_split,
        scene,
        max_steps=20000,
        unbounded=True,
        aabb=torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    ) -> None:
        # train settings
        self.max_steps = max_steps
        # dataset parameters
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        # model parameters
        self.proposal_networks = [
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=128,
            ).to(device),
            NGPDensityField(
                aabb=aabb,
                unbounded=unbounded,
                n_levels=5,
                max_resolution=256,
            ).to(device),
        ]

        self.train_dataset = DataLoader(
            subject_id=scene,
            root_fp=data_root,
            split=train_split,
            num_rays=4096,
            device=device,
            **train_dataset_kwargs,
        )

        self.test_dataset = DataLoader(
            subject_id=scene,
            root_fp=data_root,
            split="test",
            num_rays=None,
            device=device,
            **test_dataset_kwargs,
        )

        self.height, self.width, _ = self.test_dataset[0]['rays'].origins.shape
        self.render_bkgd = self.test_dataset[0]['color_bkgd']
        # setup the radiance field we want to train.
        self.prop_optimizer = torch.optim.Adam(
            itertools.chain(
                *[p.parameters() for p in self.proposal_networks],
            ),
            lr=1e-2,
            eps=1e-15,
            weight_decay=0.0,
        )
        self.prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.prop_optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.prop_optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        self.estimator = PropNetEstimator(self.prop_optimizer, self.prop_scheduler).to(device)

        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        self.radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=1e-2,
            eps=1e-15,
            weight_decay=0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[
                        max_steps // 2,
                        max_steps * 3 // 4,
                        max_steps * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

        lpips_net = LPIPS(net="vgg").to(device)
        lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        self.lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    def train(
        self,
        num_steps=16,
    ):
        # training
        for step in range(num_steps):
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
            rgb, acc, depth, extras = self.render(
                rays,
                # rendering options
                num_samples=48,
                num_samples_per_prop=[256, 9],
                near_plane=0.2,
                far_plane=1e3,
                sampling_type='lindisp',
                opaque_bkgd=True,
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

    def eval(self, rays: Rays):
        self.radiance_field.eval()
        for p in self.proposal_networks:
            p.eval()
        self.estimator.eval()

        rgb, _, _, _ = self.render(
            rays,
            num_samples=48,
            num_samples_per_prop=[256, 9],
            near_plane=0.2,
            far_plane=1e3,
            sampling_type='lindisp',
            opaque_bkgd=True,
            render_bkgd=self.render_bkgd,
        )

        return rgb

    def render(
        self,
        # scene
        rays: Rays,
        # rendering options
        num_samples: int,
        num_samples_per_prop: Sequence[int],
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        sampling_type: Literal["uniform", "lindisp"] = "lindisp",
        opaque_bkgd: bool = True,
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
            if opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[..., None, :]
            t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
                t_starts.shape[-1], dim=-2
            )
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            rgb, sigmas = self.radiance_field(positions, t_dirs)
            if opaque_bkgd:
                sigmas[..., -1, :] = torch.inf
            return rgb, sigmas.squeeze(-1)

        results = []
        chunk = (
            torch.iinfo(torch.int32).max
            if self.radiance_field.training
            else 8192
        )
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            t_starts, t_ends = self.estimator.sampling(
                prop_sigma_fns=[
                    lambda *args: prop_sigma_fn(*args, p) for p in self.proposal_networks
                ],
                prop_samples=num_samples_per_prop,
                num_samples=num_samples,
                n_rays=chunk_rays.origins.shape[0],
                near_plane=near_plane,
                far_plane=far_plane,
                sampling_type=sampling_type,
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
