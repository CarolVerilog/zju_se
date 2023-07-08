import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from utils import Rays

device = 'cuda:0'
identity=torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
]).float().to(device)


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

        self.position=torch.Tensor([0,0,0]).float()
        self.up=torch.Tensor([0,1,0]).float()
        self.right=torch.Tensor([1,0,0]).float()
        self.look=torch.Tensor([0,0,1]).float()

        self.c2w=identity
        self.view_dirty=True
        self.update()

    def update(self):
        if(self.view_dirty):
            self.look = F.normalize(self.look, p=2, dim=0)
            self.up = F.normalize(torch.cross(self.look, self.right), p=2, dim=0)
            self.right = torch.cross(self.up, self.look)

            ori_x=torch.dot(self.position, self.right)
            ori_y=torch.dot(self.position, self.up)
            ori_z=torch.dot(self.position, self.look)

            w2c=torch.Tensor([
                [*self.right, ori_x],
                [*self.up, ori_y],
                [*self.look, ori_z],
                [0,0,0,1]
            ])
            self.c2w=torch.inverse(w2c).to(device)

            view_dirs = (self.camera_dirs[:, :, None, :] * self.c2w[None, :3, :3]).sum(dim=-1)
            origins = torch.broadcast_to(self.c2w[:3, -1], view_dirs.shape)
            self.rays = Rays(origins=origins, viewdirs=view_dirs)
            self.view_dirty=False

    def walk(self, d):
        self.position+=self.look*d
        self.view_dirty=True

    def strafe(self, d):
        self.position+=self.right*d
        self.view_dirty=True

    def pitch(self, angle):
        rotation=torch.Tensor(R.from_rotvec(self.right, angle))
        self.up=rotation@self.up
        self.look=rotation@self.look
        self.view_dirty=True

    def roll(self, angle):
        rotation=torch.Tensor(R.from_rotvec(self.look, angle))
        self.right=rotation@self.right
        self.up=rotation@self.up
        self.view_dirty=True

    def yaw(self, angle):
        rotation=torch.Tensor(R.from_rotvec(self.up, angle))
        self.right=rotation@self.right
        self.look=rotation@self.look
        self.view_dirty=True
