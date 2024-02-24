import torch
from IPython.display import display
from ipywidgets import FloatSlider, Dropdown, HBox, VBox, Text, Button
from PIL import Image
import ipywidgets as widgets
import io
import os
from .BaseUI import BaseUI

class Img2ImgRefinerUI:
    def __init__(self, ui: BaseUI):
        self.__generator = torch.Generator(device="cuda")
        self.__base_ui = ui
        self.__current_file = ""
        
        self.strength_field = FloatSlider(value=0.35, min=0, max=1, step=0.05, description="Strength: ")
        self.upscale_field = FloatSlider(value=1, min=1, max=4, step=0.5, description="Upscale factor: ")
        self.path_field = Text(description="Url:", placeholder="Image path...")
        self.load_button = Button(description='Choose')
        self.image_preview = widgets.Image()
        
        def load_handler(b):
            if not os.path.isfile(self.path_field.value): return
            self.__current_file = self.path_field.value
            self.image_preview.value = open(self.__current_file, "rb").read()
        self.load_button.on_click(load_handler)
        
    def generate(self, pipe, init_image=None, generator=None):
        """Generate images given Img2Img Pipeline, and settings set in Base UI and Refiner UI."""
        if self.__base_ui.seed_field.value >= 0: 
            seed = self.__base_ui.seed_field.value
        else:
            seed = self.__generator.seed()
            
        if init_image is None:
            if not os.path.isfile(self.__current_file): return
            init_image = Image.open(self.__current_file)
        
        init_image = init_image.convert('RGB')
        size = (int(self.upscale_field.value * init_image.size[0]), int(self.upscale_field.value * init_image.size[1]))
        init_image = init_image.resize(size, resample=Image.LANCZOS)
        
        g = torch.cuda.manual_seed(seed)
        self._metadata = self.__base_ui.get_metadata_string() + f"\nImg2Img Seed: {seed}, Noise Strength: {self.strength_field.value}, Upscale: {self.upscale_field.value} "
        
        results = pipe(image=init_image,
                       prompt=self.__base_ui.positive_prompt.value, 
                       negative_prompt=self.__base_ui.negative_prompt.value, 
                       num_inference_steps=self.__base_ui.steps_field.value,
                       num_images_per_prompt = self.__base_ui.batch_field.value,
                       guidance_scale=self.__base_ui.cfg_field.value, 
                       strength=self.strength_field.value,
                       generator=g)
        return results

    @property
    def metadata(self):
        return self._metadata 

    @property
    def render_element(self): 
        return VBox([self.strength_field, self.upscale_field, HBox([self.path_field, self.load_button]), self.image_preview])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()