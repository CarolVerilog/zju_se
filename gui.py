import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import dearpygui.dearpygui as dpg

from utils import Rays
from nerfacto import NeRFacto

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
    def __init__(self, width, height) -> None:
        x, y = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )
        x = x.to(device)
        y = y.to(device)

        self.camera_dirs = F.pad(
            torch.stack(
                [
                    (x + 0.5 - width // 2) / (width // 2),
                    (y + 0.5 - height // 2) / (height // 2),
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
        self.down = torch.Tensor([0, 1, 0]).float()
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
        rotation = torch.Tensor(
            R.from_rotvec(self.right * np.deg2rad(-angle)).as_matrix()
        )
        self.down = rotation @ self.down
        self.look = rotation @ self.look
        self.view_dirty = True

    def roll(self, angle):
        rotation = torch.Tensor(
            R.from_rotvec(self.look * np.deg2rad(angle)).as_matrix()
        )
        self.right = rotation @ self.right
        self.down = rotation @ self.down
        self.view_dirty = True

    def yaw(self, angle):
        rotation = torch.Tensor(
            R.from_rotvec(self.down * np.deg2rad(angle)).as_matrix()
        )
        self.right = rotation @ self.right
        self.look = rotation @ self.look
        self.view_dirty = True

    def update(self):
        if self.view_dirty:
            self.look = F.normalize(self.look, p=2, dim=0)
            self.down = F.normalize(torch.cross(self.look, self.right), p=2, dim=0)
            self.right = torch.cross(self.down, self.look)

            ori_x = -torch.dot(self.position, self.right)
            ori_y = -torch.dot(self.position, self.down)
            ori_z = -torch.dot(self.position, self.look)

            w2c = torch.Tensor(
                [
                    [*self.right, ori_x],
                    [*self.down, ori_y],
                    [*self.look, ori_z],
                    [0, 0, 0, 1],
                ]
            )
            self.c2w = torch.inverse(w2c).to(device)

            view_dirs = (self.camera_dirs[:, :, None, :] * self.c2w[None, :3, :3]).sum(
                dim=-1
            )
            origins = torch.broadcast_to(self.c2w[:3, -1], view_dirs.shape)
            self.rays = Rays(origins=origins, viewdirs=view_dirs)
            self.view_dirty = False


class GUI:
    def __init__(self, width, height) -> None:
        self.nerf = NeRFacto()
        self.width = width
        self.height = height
        self.camera = Camera(self.width, self.height)
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.start = False
        self.training = False

        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="texture",
            )

        def A_down(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            self.camera.strafe(-0.01)

        def S_down(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            self.camera.walk(-0.01)

        def W_down(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            self.camera.walk(+0.01)

        def D_down(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            self.camera.strafe(+0.01)

        def mouse_drag(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            dx, dy = app_data[1] - self.last_mouse_x, app_data[2] - self.last_mouse_y
            self.last_mouse_x, self.last_mouse_y = app_data[1], app_data[2]
            self.camera.yaw(dx * 0.05)
            self.camera.pitch(dy * 0.05)

        def mouse_release(sender, app_data):
            if not dpg.is_item_focused("primary_window"):
                return
            self.last_mouse_x = self.last_mouse_y = 0

        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_A, callback=A_down)
            dpg.add_key_down_handler(dpg.mvKey_S, callback=S_down)
            dpg.add_key_down_handler(dpg.mvKey_W, callback=W_down)
            dpg.add_key_down_handler(dpg.mvKey_D, callback=D_down)
            dpg.add_mouse_drag_handler(dpg.mvMouseButton_Left, callback=mouse_drag)
            dpg.add_mouse_release_handler(
                dpg.mvMouseButton_Left, callback=mouse_release
            )

        with dpg.window(
            label="Control",
            tag="control_window",
            width=self.width // 3,
            height=self.height // 2,
        ):
            with dpg.collapsing_header(label="Train", default_open=True):
                with dpg.collapsing_header(label="Static", default_open=True):
                    with dpg.group(horizontal=False):

                        def callback_data_root(sender, appdata):
                            self.nerf.data_root = appdata

                        dpg.add_input_text(
                            label="Data root",
                            tag="data_root",
                            default_value="data",
                            callback=callback_data_root,
                        )

                        def callback_train_split(sender, appdata):
                            self.nerf.train_split = appdata

                        dpg.add_input_text(
                            label="Train split",
                            tag="train_split",
                            default_value="train",
                            callback=callback_train_split,
                        )

                        def callback_scene(sender, appdata):
                            self.nerf.scene = appdata

                        dpg.add_input_text(
                            label="Scene",
                            tag="scene",
                            default_value="garden",
                            callback=callback_scene,
                        )

                        def callback_max_steps(sender, appdata):
                            self.nerf.max_steps = appdata

                        dpg.add_input_int(
                            label="Max steps",
                            tag="max_steps",
                            default_value=20000,
                            callback=callback_max_steps,
                        )

                        def callback_lr(sender, appdata):
                            self.nerf.lr = appdata

                        dpg.add_input_float(
                            label="Learning rate",
                            tag="lr",
                            default_value=1e-2,
                            callback=callback_lr,
                        )

                        def callback_eps(sender, appdata):
                            self.nerf.eps = appdata

                        dpg.add_input_float(
                            label="Epsilon",
                            tag="eps",
                            default_value=1e-15,
                            callback=callback_eps,
                        )

                        def callback_weight_decay(sender, appdata):
                            self.nerf.weight_decay = appdata

                        dpg.add_input_float(
                            label="Weight decay",
                            tag="weight_decay",
                            default_value=0.0,
                            callback=callback_weight_decay,
                        )

                        def callback_train_num_rays(sender, appdata):
                            self.nerf.train_num_rays = appdata

                        dpg.add_input_int(
                            label="Rays",
                            tag="train_num_rays",
                            default_value=4096,
                            callback=callback_train_num_rays,
                        )

                with dpg.collapsing_header(label="Dynamic", default_open=True):
                    with dpg.group(horizontal=False):

                        def callback_train_num_samples(sender, appdata):
                            self.nerf.train_num_samples = appdata

                        dpg.add_input_int(
                            label="Ray samples",
                            tag="train_num_samples",
                            default_value=48,
                            callback=callback_train_num_samples,
                        )

                        def callback_train_num_samples_prop0(sender, appdata):
                            self.nerf.train_num_samples_per_prop[0] = appdata

                        dpg.add_input_int(
                            label="Prop 0 samples",
                            tag="train_num_samples_prop0",
                            default_value=256,
                            callback=callback_train_num_samples_prop0,
                        )

                        def callback_train_num_samples_prop1(sender, appdata):
                            self.nerf.train_num_samples_per_prop[1] = appdata

                        dpg.add_input_int(
                            label="Prop 1 samples",
                            tag="train_num_samples_prop1",
                            default_value=9,
                            callback=callback_train_num_samples_prop1,
                        )

                        def callback_train_near_plane(sender, appdata):
                            self.nerf.train_near_plane = appdata

                        dpg.add_input_float(
                            label="Near plane",
                            tag="train_near_plane",
                            default_value=0.2,
                            callback=callback_train_near_plane,
                        )

                        def callback_train_far_plane(sender, appdata):
                            self.nerf.train_far_plane = appdata

                        dpg.add_input_float(
                            label="Far plane",
                            tag="train_far_plane",
                            default_value=1e3,
                            callback=callback_train_far_plane,
                        )

            with dpg.collapsing_header(label="Render", default_open=True):

                def callback_render_num_samples(sender, appdata):
                    self.nerf.render_num_samples = appdata

                dpg.add_input_int(
                    label="Ray samples",
                    tag="render_num_samples",
                    default_value=16,
                    callback=callback_render_num_samples,
                )

                def callback_render_num_samples_prop0(sender, appdata):
                    self.nerf.render_num_samples_per_prop[0] = appdata

                dpg.add_input_int(
                    label="Prop 0 samples",
                    tag="render_num_samples_prop0",
                    default_value=16,
                    callback=callback_render_num_samples_prop0,
                )

                def callback_render_num_samples_prop1(sender, appdata):
                    self.nerf.render_num_samples_per_prop[1] = appdata

                dpg.add_input_int(
                    label="Prop 1 samples",
                    tag="render_num_samples_prop1",
                    default_value=9,
                    callback=callback_render_num_samples_prop1,
                )

                def callback_render_near_plane(sender, appdata):
                    self.nerf.render_near_plane = appdata

                dpg.add_input_float(
                    label="Near plane",
                    tag="render_near_plane",
                    default_value=0.2,
                    callback=callback_render_near_plane,
                )

                def callback_render_far_plane(sender, appdata):
                    self.nerf.render_far_plane = appdata

                dpg.add_input_float(
                    label="Far plane",
                    tag="render_far_plane",
                    default_value=1e3,
                    callback=callback_render_far_plane,
                )

                def callback_render_chunk(sender, appdata):
                    self.nerf.render_chunk = appdata

                dpg.add_input_int(
                    label="Rays chunk",
                    tag="render_chunk",
                    default_value=8192,
                    callback=callback_render_chunk,
                )

            def callback_train(sender, app_data):
                if self.start == False:
                    dpg.disable_item("data_root")
                    dpg.disable_item("train_split")
                    dpg.disable_item("scene")
                    dpg.disable_item("max_steps")
                    dpg.disable_item("lr")
                    dpg.disable_item("eps")
                    dpg.disable_item("weight_decay")
                    dpg.disable_item("train_num_rays")
                    self.nerf.populate()
                    self.start = True
                if self.training:
                    self.training = False
                    dpg.configure_item("button_train", label="start")
                else:
                    self.training = True
                    dpg.configure_item("button_train", label="stop")

            dpg.add_button(label="start", tag="button_train", callback=callback_train)

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
            self.camera.update()

            if self.training and total_step < self.nerf.max_steps:
                self.nerf.train(num_step)
                total_step += num_step

            if self.start:
                with torch.no_grad():
                    pass
                    rgb = self.nerf.eval(self.camera.rays)
                    dpg.set_value("texture", rgb.cpu().numpy().reshape(-1))

            dpg.render_dearpygui_frame()
