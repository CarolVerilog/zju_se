import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg
from utils import Rays

device = "cuda:0"
identity = (
    torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    .float()
    .to(device)
)


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

        self.position = torch.Tensor([0, 0, 0]).float()
        self.up = torch.Tensor([0, 1, 0]).float()
        self.right = torch.Tensor([1, 0, 0]).float()
        self.look = torch.Tensor([0, 0, 1]).float()

        self.c2w = identity
        self.view_dirty = True
        self.update()

    def walk(self, d):
        self.position += self.look * d
        self.view_dirty = True

    def strafe(self, d):
        self.position += self.right * d
        self.view_dirty = True

    def pitch(self, angle):
        rotation = torch.Tensor(R.from_rotvec(self.right, angle))
        self.up = rotation @ self.up
        self.look = rotation @ self.look
        self.view_dirty = True

    def roll(self, angle):
        rotation = torch.Tensor(R.from_rotvec(self.look, angle))
        self.right = rotation @ self.right
        self.up = rotation @ self.up
        self.view_dirty = True

    def yaw(self, angle):
        rotation = torch.Tensor(R.from_rotvec(self.up, angle))
        self.right = rotation @ self.right
        self.look = rotation @ self.look
        self.view_dirty = True

    def update(self):
        if self.view_dirty:
            self.look = F.normalize(self.look, p=2, dim=0)
            self.up = F.normalize(torch.cross(self.look, self.right), p=2, dim=0)
            self.right = torch.cross(self.up, self.look)

            ori_x = torch.dot(self.position, self.right)
            ori_y = torch.dot(self.position, self.up)
            ori_z = torch.dot(self.position, self.look)

            w2c = torch.Tensor(
                [
                    [*self.right, ori_x],
                    [*self.up, ori_y],
                    [*self.look, ori_z],
                    [0, 0, 0, 1],
                ]
            )
            self.c2w = torch.inverse(w2c).to(device)
            print(self.c2w)

            view_dirs = (self.camera_dirs[:, :, None, :] * self.c2w[None, :3, :3]).sum(
                dim=-1
            )
            origins = torch.broadcast_to(self.c2w[:3, -1], view_dirs.shape)
            self.rays = Rays(origins=origins, viewdirs=view_dirs)
            self.view_dirty = False


class GUI:
    def __init__(self, nerf) -> None:
        self.nerf = nerf
        self.height = nerf.height
        self.width = nerf.width
        self.camera = Camera(nerf.K, self.height, self.width)
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)

        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="texture",
            )

        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_W, callback=self.camera.walk(+5.0))
            dpg.add_key_down_handler(dpg.mvKey_S, callback=S_down)
            dpg.add_key_down_handler(dpg.mvKey_D, callback=self.camera.strafe(+5.0))
            dpg.add_key_down_handler(dpg.mvKey_A, callback=self.camera.strafe(-5.0))

        with dpg.window(tag="primary_window", width=self.width, height=self.height):
            dpg.add_image("texture")
            dpg.set_primary_window("primary_window", True)

        dpg.create_viewport(
            title="nerfacto", width=self.width, height=self.height, resizable=False
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        num_step = 10
        total_step = 0

        while dpg.is_dearpygui_running():
            if total_step < self.nerf.max_steps:
                self.nerf.train(num_step)
                total_step += num_step

            with torch.no_grad():
                self.camera.update()
                rgb = self.nerf.eval(self.camera.rays)

                dpg.set_value("texture", rgb.cpu().numpy().reshape(-1))

            dpg.render_dearpygui_frame()
