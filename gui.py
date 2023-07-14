import time
import datetime
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import dearpygui.dearpygui as dpg
import imageio
import tqdm

from utils import Rays
from distortion_ngp import DistortionNGP

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
            R.from_rotvec(self.right * np.deg2rad(angle)).as_matrix()
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

    def rotate(self, axis, angle):
        rotation = torch.Tensor(R.from_rotvec(axis * np.deg2rad(angle)).as_matrix())
        self.down = rotation @ self.down
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
        self.nerf = DistortionNGP()
        self.width = width
        self.height = height
        self.camera = Camera(self.width, self.height)
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.start = False
        self.training = False
        self.drawing = False
        self.output_dir = "output"
        self.test_video_radius = 0.6
        self.test_video_pitch = -45.0
        self.test_video_length = 4.0
        self.test_video_width = 1024
        self.test_video_height = 1024

        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="render_buffer",
            )

        def A_down(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
                return
            self.camera.strafe(-0.01)

        def S_down(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
                return
            self.camera.walk(-0.01)

        def W_down(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
                return
            self.camera.walk(+0.01)

        def D_down(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
                return
            self.camera.strafe(+0.01)

        def mouse_drag(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
                return
            dx, dy = app_data[1] - self.last_mouse_x, app_data[2] - self.last_mouse_y
            self.last_mouse_x, self.last_mouse_y = app_data[1], app_data[2]
            self.camera.yaw(dx * 0.05)
            self.camera.pitch(-dy * 0.05)

        def mouse_release(sender, app_data):
            if (
                not dpg.is_item_focused("primary_window")
                or not self.drawing
                or not self.start
            ):
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
            modal=True,
            show=False,
            tag="modal",
            no_title_bar=True,
            no_move=True,
            width=width // 4,
            height=height // 4,
            pos=[width // 2 - width // 8, height // 2 - height // 8],
        ):
            dpg.add_text("", tag="modal_text")

        with dpg.window(
            label="Control",
            tag="control_window",
            width=self.width // 3,
            height=self.height // 2,
        ):
            dpg.add_text(default_value=f"FPS: {0}, MSPF: {0.0}", tag="fps")
            with dpg.collapsing_header(label="Train", default_open=False):
                dpg.add_text(
                    default_value=f"Iter: {0}/{self.nerf.max_steps}", tag="iter"
                )
                dpg.add_text(
                    default_value="Training time: 00:00:000", tag="training_time"
                )

                with dpg.plot(
                    label="Loss",
                    width=-1,
                    height=self.height // 4,
                ):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="iter", tag="iter_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="loss", tag="loss_axis")
                    dpg.add_line_series([], [], tag="loss_series", parent="loss_axis")

                def callback_data_root(sender, appdata):
                    self.nerf.data_root = appdata

                dpg.add_input_text(
                    label="Data root",
                    tag="data_root",
                    default_value=self.nerf.data_root,
                    callback=callback_data_root,
                )

                def callback_train_split(sender, appdata):
                    self.nerf.train_split = appdata

                dpg.add_input_text(
                    label="Train split",
                    tag="train_split",
                    default_value=self.nerf.train_split,
                    callback=callback_train_split,
                )

                def callback_scene(sender, appdata):
                    self.nerf.scene = appdata

                dpg.add_input_text(
                    label="Scene",
                    tag="scene",
                    default_value=self.nerf.scene,
                    callback=callback_scene,
                )

                def callback_factor(sender, appdata):
                    self.nerf.factor = appdata

                dpg.add_input_int(
                    label="Factor",
                    tag="factor",
                    default_value=self.nerf.factor,
                    callback=callback_factor,
                )

                def callback_max_steps(sender, appdata):
                    self.nerf.max_steps = appdata
                    dpg.set_value("iter", f"Iter: {0}/{self.nerf.max_steps}")

                dpg.add_input_int(
                    label="Max steps",
                    tag="max_steps",
                    default_value=self.nerf.max_steps,
                    callback=callback_max_steps,
                )

                def callback_lr(sender, appdata):
                    self.nerf.lr = appdata

                dpg.add_input_float(
                    label="Learning rate",
                    tag="lr",
                    default_value=self.nerf.lr,
                    callback=callback_lr,
                )

                def callback_eps(sender, appdata):
                    self.nerf.eps = appdata

                dpg.add_input_float(
                    label="Epsilon",
                    tag="eps",
                    default_value=self.nerf.eps,
                    callback=callback_eps,
                )

                def callback_weight_decay(sender, appdata):
                    self.nerf.weight_decay = appdata

                dpg.add_input_float(
                    label="Weight decay",
                    tag="weight_decay",
                    default_value=self.nerf.weight_decay,
                    callback=callback_weight_decay,
                )

                def callback_train_num_rays(sender, appdata):
                    self.nerf.train_num_rays = appdata

                dpg.add_input_int(
                    label="Rays",
                    tag="train_num_rays",
                    default_value=self.nerf.train_num_rays,
                    callback=callback_train_num_rays,
                )

                def callback_train_num_samples(sender, appdata):
                    self.nerf.train_num_samples = appdata

                dpg.add_input_int(
                    label="Ray samples",
                    tag="train_num_samples",
                    default_value=self.nerf.train_num_samples,
                    callback=callback_train_num_samples,
                )

                def callback_train_num_samples_prop0(sender, appdata):
                    self.nerf.train_num_samples_per_prop[0] = appdata

                dpg.add_input_int(
                    label="Prop 0 samples",
                    tag="train_num_samples_prop0",
                    default_value=self.nerf.train_num_samples_per_prop[0],
                    callback=callback_train_num_samples_prop0,
                )

                def callback_train_num_samples_prop1(sender, appdata):
                    self.nerf.train_num_samples_per_prop[1] = appdata

                dpg.add_input_int(
                    label="Prop 1 samples",
                    tag="train_num_samples_prop1",
                    default_value=self.nerf.train_num_samples_per_prop[1],
                    callback=callback_train_num_samples_prop1,
                )

                def callback_train_near_plane(sender, appdata):
                    self.nerf.train_near_plane = appdata

                dpg.add_input_float(
                    label="Near plane",
                    tag="train_near_plane",
                    default_value=self.nerf.train_near_plane,
                    callback=callback_train_near_plane,
                )

                def callback_train_far_plane(sender, appdata):
                    self.nerf.train_far_plane = appdata

                dpg.add_input_float(
                    label="Far plane",
                    tag="train_far_plane",
                    default_value=self.nerf.train_far_plane,
                    callback=callback_train_far_plane,
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

                        dpg.set_value("modal_text", "Loading data...")
                        dpg.configure_item("modal", show=True)
                        self.nerf.populate()
                        dpg.configure_item("modal", show=False)

                        self.start = True

                    if self.training:
                        self.training = False
                        dpg.configure_item("button_train", label="continue")
                    else:
                        self.training = True
                        dpg.configure_item("button_train", label="pause")

                dpg.add_button(
                    label="Train", tag="button_train", callback=callback_train
                )

            with dpg.collapsing_header(label="Test", default_open=False):

                def callback_output_dir(sender, appdata):
                    self.output_dir = appdata

                dpg.add_input_text(
                    label="Output directory",
                    tag="output_dir",
                    default_value=self.output_dir,
                    callback=callback_output_dir,
                )

                def callback_test_video_radius(sender, appdata):
                    self.test_video_radius = appdata

                dpg.add_input_float(
                    label="Video radius",
                    tag="test_video_radius",
                    default_value=self.test_video_radius,
                    callback=callback_test_video_radius,
                )

                def callback_test_video_pitch(sender, appdata):
                    self.test_video_pitch = appdata

                dpg.add_input_float(
                    label="Video pitch",
                    tag="test_video_pitch",
                    default_value=self.test_video_pitch,
                    callback=callback_test_video_pitch,
                )

                def callback_test_video_length(sender, appdata):
                    self.test_video_length = appdata

                dpg.add_input_float(
                    label="Video length",
                    tag="test_video_length",
                    default_value=self.test_video_length,
                    callback=callback_test_video_length,
                )

                def callback_test_video_width(sender, appdata):
                    self.test_video_width = appdata

                dpg.add_input_int(
                    label="Video width",
                    tag="test_video_width",
                    default_value=self.test_video_width,
                    callback=callback_test_video_width,
                )

                def callback_test_video_height(sender, appdata):
                    self.test_video_height = appdata

                dpg.add_input_int(
                    label="Video height",
                    tag="test_video_height",
                    default_value=self.test_video_height,
                    callback=callback_test_video_height,
                )

                def callback_test_num_samples(sender, appdata):
                    self.nerf.test_num_samples = appdata

                dpg.add_input_int(
                    label="Ray samples",
                    tag="test_num_samples",
                    default_value=self.nerf.test_num_samples,
                    callback=callback_test_num_samples,
                )

                def callback_test_num_samples_prop0(sender, appdata):
                    self.nerf.test_num_samples_per_prop[0] = appdata

                dpg.add_input_int(
                    label="Prop 0 samples",
                    tag="test_num_samples_prop0",
                    default_value=self.nerf.test_num_samples_per_prop[0],
                    callback=callback_test_num_samples_prop0,
                )

                def callback_test_num_samples_prop1(sender, appdata):
                    self.nerf.test_num_samples_per_prop[1] = appdata

                dpg.add_input_int(
                    label="Prop 1 samples",
                    tag="test_num_samples_prop1",
                    default_value=self.nerf.test_num_samples_per_prop[1],
                    callback=callback_test_num_samples_prop1,
                )

                def callback_test_near_plane(sender, appdata):
                    self.nerf.test_near_plane = appdata

                dpg.add_input_float(
                    label="Near plane",
                    tag="test_near_plane",
                    default_value=self.nerf.test_near_plane,
                    callback=callback_test_near_plane,
                )

                def callback_test_far_plane(sender, appdata):
                    self.nerf.test_far_plane = appdata

                dpg.add_input_float(
                    label="Far plane",
                    tag="test_far_plane",
                    default_value=self.nerf.test_far_plane,
                    callback=callback_test_far_plane,
                )

                def callback_test_chunk_size(sender, appdata):
                    self.nerf.test_chunk_size = appdata

                dpg.add_input_int(
                    label="Rays chunk size",
                    tag="test_chunk_size",
                    default_value=self.nerf.test_chunk_size,
                    callback=callback_test_chunk_size,
                )

                def callback_test(sender, app_data):
                    if self.start == False:
                        return

                    dpg.set_value("modal_text", "Testing NeRF...")
                    dpg.configure_item("modal", show=True)

                    training_flag = self.training
                    if training_flag == True:
                        self.training = False

                    if not os.path.isdir(self.output_dir):
                        os.mkdir(self.output_dir)

                    file_name = (
                        self.output_dir
                        + "/"
                        + self.nerf.scene
                        + "_radius_"
                        + str(self.test_video_radius)
                        + "_pitch_"
                        + str(self.test_video_pitch)
                        + "_"
                        + datetime.datetime.strftime(
                            datetime.datetime.now(), "%y-%m-%d_%H:%M:%S"
                        )
                    )
                    json_data = {}

                    try:
                        ssim, psnr, lpips = self.nerf.test()
                    except:
                        dpg.configure_item("modal", show=False)
                        return

                    json_data["ssim"] = ssim
                    json_data["psnr"] = psnr
                    json_data["lpips"] = lpips
                    json_data["iter"] = self.nerf.max_steps
                    json_data["time"] = dpg.get_value("training_time").split(": ")[1]

                    json_file = open(file_name + ".json", "w")
                    json.dump(json_data, json_file)

                    rgbs = []
                    print("Rendering video")

                    video_camera = Camera(self.test_video_width, self.test_video_height)
                    frames = int(30 * self.test_video_length)
                    angle = 360 / frames

                    for _ in tqdm.tqdm(range(frames)):
                        video_camera.yaw(angle)
                        video_camera.pitch(self.test_video_pitch)
                        video_camera.walk(-self.test_video_radius)
                        video_camera.update()

                        try:
                            rgb = self.nerf.eval(video_camera.rays).cpu()
                        except:
                            dpg.configure_item("modal", show=False)
                            return

                        rgbs.append(rgb)
                        video_camera.walk(self.test_video_radius)
                        video_camera.pitch(-self.test_video_pitch)

                    rgbs = (torch.stack(rgbs, 0).numpy() * 255).astype(np.uint8)
                    imageio.mimwrite(
                        file_name + ".mp4",
                        rgbs,
                        fps=30,
                        quality=10,
                        macro_block_size=None,
                    )

                    if training_flag == True:
                        self.training = True

                    dpg.configure_item("modal", show=False)

                dpg.add_button(label="Test", tag="button_test", callback=callback_test)

            with dpg.collapsing_header(label="Draw", default_open=False):

                def callback_draw_num_samples(sender, appdata):
                    self.nerf.draw_num_samples = appdata

                dpg.add_input_int(
                    label="Ray samples",
                    tag="draw_num_samples",
                    default_value=self.nerf.draw_num_samples,
                    callback=callback_draw_num_samples,
                )

                def callback_draw_num_samples_prop0(sender, appdata):
                    self.nerf.draw_num_samples_per_prop[0] = appdata

                dpg.add_input_int(
                    label="Prop 0 samples",
                    tag="draw_num_samples_prop0",
                    default_value=self.nerf.draw_num_samples_per_prop[0],
                    callback=callback_draw_num_samples_prop0,
                )

                def callback_draw_num_samples_prop1(sender, appdata):
                    self.nerf.draw_num_samples_per_prop[1] = appdata

                dpg.add_input_int(
                    label="Prop 1 samples",
                    tag="draw_num_samples_prop1",
                    default_value=self.nerf.draw_num_samples_per_prop[1],
                    callback=callback_draw_num_samples_prop1,
                )

                def callback_draw_near_plane(sender, appdata):
                    self.nerf.draw_near_plane = appdata

                dpg.add_input_float(
                    label="Near plane",
                    tag="draw_near_plane",
                    default_value=self.nerf.draw_near_plane,
                    callback=callback_draw_near_plane,
                )

                def callback_draw_far_plane(sender, appdata):
                    self.nerf.draw_far_plane = appdata

                dpg.add_input_float(
                    label="Far plane",
                    tag="draw_far_plane",
                    default_value=self.nerf.draw_far_plane,
                    callback=callback_draw_far_plane,
                )

                def callback_draw_chunk_size(sender, appdata):
                    self.nerf.draw_chunk_size = appdata

                dpg.add_input_int(
                    label="Rays chunk size",
                    tag="draw_chunk_size",
                    default_value=self.nerf.draw_chunk_size,
                    callback=callback_draw_chunk_size,
                )

                def callback_draw(sender, appdata):
                    if not self.start:
                        return
                    self.drawing = not self.drawing

                dpg.add_checkbox(
                    label="Draw", tag="checkbox_draw", callback=callback_draw
                )

        with dpg.window(tag="primary_window", width=self.width, height=self.height):
            dpg.add_image("render_buffer")
            dpg.set_primary_window("primary_window", True)

        dpg.create_viewport(
            title="nerfacto",
            width=self.width,
            height=self.height,
            resizable=False,
        )
        dpg.setup_dearpygui()
        dpg.maximize_viewport()
        dpg.show_viewport()

    def run(self):
        step = 0
        training_time = 0
        frame_count = 0
        elapsed_time = 0.0
        iters = []
        losses = []

        while dpg.is_dearpygui_running():
            frame_time = time.time()

            if self.start:
                if self.training and step < self.nerf.max_steps:
                    iters.append(step)

                    starter, ender = torch.cuda.Event(
                        enable_timing=True
                    ), torch.cuda.Event(enable_timing=True)
                    starter.record()

                    losses.append(self.nerf.train(step))

                    ender.record()
                    torch.cuda.synchronize()
                    training_time += int(starter.elapsed_time(ender))

                    secs = training_time // 1000
                    m = secs // 60
                    s = secs % 60
                    ms = training_time % 1000

                    dpg.set_value(
                        "training_time",
                        f"Training time: {m:02d}:{s:02d}:{ms:03d}",
                    )

                    step += 1
                    dpg.set_value("iter", f"Iter: {step}/{self.nerf.max_steps}")
                    dpg.set_value("loss_series", [iters, losses])

                    plot_iters_left = max(0, step - 1000)
                    plot_iters_right = step
                    plot_losses = losses[plot_iters_left:plot_iters_right]
                    dpg.set_axis_limits("iter_axis", plot_iters_left, plot_iters_right)
                    dpg.set_axis_limits("loss_axis", min(plot_losses), max(plot_losses))

                    if step >= self.nerf.max_steps:
                        dpg.configure_item("button_train", label="finished")
                        dpg.disable_item("button_train")
                else:
                    dpg.set_axis_limits_auto("iter_axis")
                    dpg.set_axis_limits_auto("loss_axis")

                if self.drawing:
                    self.camera.update()
                    rgb = self.nerf.eval(self.camera.rays)
                    dpg.set_value("render_buffer", rgb.cpu().numpy().reshape(-1))

            dpg.render_dearpygui_frame()

            frame_time = time.time() - frame_time
            elapsed_time += frame_time
            frame_count += 1

            if elapsed_time >= 1.0:
                fps = frame_count if frame_count > 1 else 1 / elapsed_time
                mspf = 1000 / fps
                dpg.set_value("fps", f"FPS: {fps:.2f} MSPF: {mspf:.2f}")
                elapsed_time = 0.0
                frame_count = 0
