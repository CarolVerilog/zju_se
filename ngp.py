from typing import Callable, List, Union

import numpy as np
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn


class TruncExp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = lambda x: TruncExp.apply(x - 1)


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    return x


class NGPRadianceField(torch.nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        num_dim: int = 3,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                ],
            },
        )

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=(
                self.direction_encoding.n_output_dims
                + self.geo_feat_dim
            ),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def query_density(self, positions, return_feat: bool = False):
        positions = contract_to_unisphere(positions, self.aabb)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = (
            self.mlp_base(positions.view(-1, self.num_dim))
            .view(list(positions.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(positions)
        )
        density_before_activation, base_mlp_out = torch.split(
            positions, [1, self.geo_feat_dim], dim=-1
        )
        density = trunc_exp(density_before_activation) * selector[..., None]
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def query_rgb(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        dir = (dir + 1.0) / 2.0
        d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.mlp_head(h).reshape(list(embedding.shape[:-1]) + [3]).to(embedding)
        rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self.query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore


class NGPDensityField(torch.nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        num_dim: int = 3,
        base_resolution: int = 16,
        max_resolution: int = 128,
        n_levels: int = 5,
        log2_hashmap_size: int = 17,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

    def forward(self, positions: torch.Tensor):
        positions = contract_to_unisphere(positions, self.aabb)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        density_before_activation = (
            self.mlp_base(positions.view(-1, self.num_dim))
            .view(list(positions.shape[:-1]) + [1])
            .to(positions)
        )
        density = trunc_exp(density_before_activation) * selector[..., None]
        return density
