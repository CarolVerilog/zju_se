import numpy as np
import torch
import dearpygui.dearpygui as dpg

from camera import Camera

class GUI:
    def __init__(self, nerf) -> None:
        self.nerf = nerf
        self.height = nerf.height
        self.width=nerf.width
        self.camera = Camera(nerf.train_dataset.K, self.height, self.width)
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)

        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="texture")

        with dpg.window(tag="primary_window", width=self.width, height=self.height):
            dpg.add_image("texture")

        dpg.set_primary_window("primary_window", True)

        dpg.create_viewport(title='nerfacto', width=self.width, height=self.height, resizable=False)
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
                rgb = self.nerf.eval(self.camera.rays)

                dpg.set_value("texture", rgb.cpu().numpy().reshape(-1))

            dpg.render_dearpygui_frame()
