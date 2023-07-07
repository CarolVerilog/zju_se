import numpy as np
import torch
import torch.nn.functional as F

from utils import Rays

device = 'cuda:0'

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

class Camera:
    def __init__(self, K, H, W) -> None:

        x, y = torch.meshgrid(
                torch.arange(W),
                torch.arange(H),
                indexing="xy",
            )
        x = x.to(device)
        y = y.to(device)

        self.camera_dirs = F.pad(
                torch.stack(
                    [
                        (x - K[0, 2] + 0.5) / K[0, 0],
                        (y - K[1, 2] + 0.5) / K[1, 1],
                    ],
                    dim=-1,
                ),
                (0, 1),
                value=1.0,
            )

        self.camera_dirs = self.camera_dirs / torch.linalg.norm(
                self.camera_dirs, dim=-1, keepdims=True
            )

        self.c2w=trans_t(0).to(device)

    def get_rays(self):
        view_dirs = (self.camera_dirs[:, :, None, :] * self.c2w[None, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(self.c2w[:3, -1], view_dirs.shape)

        rays = Rays(origins=origins, viewdirs=view_dirs)
        return rays

class GUI:
    def __init__(self, nerf, H, W) -> None:
        self.nerf=nerf
        self.camera=Camera(nerf.train_dataset.K, H, W)

    def train(self):
        self.nerf.train(20000)


