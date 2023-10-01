#!/usr/bin/env python

#!/usr/bin/env python3

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"

#model_id = "Envvi/Inkpunk-Diffusion"
model_id = "stabilityai/stable-diffusion-2-1"

#prompt = "nvinkpunk A perfect cartoon drawing of the given photograph in an anime style" 
#prompt = "A perfect cartoon drawing of the given photograph in disney style" 
prompt = "Stylized image"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None, use_safetensors=False, torch_dtype=torch.float16, variant="fp16").to(device)

init_image_file = input("Enter sketch image file: ")
init_image = Image.open(init_image_file).convert("RGB") 

width, height = init_image.size
new_height = int(768 * height / width)
init_image = init_image.resize((768, new_height))

output_file = input("Enter output file name: ")

images = pipe(prompt=prompt, image=init_image, strength=0.5, guidance_scale=12).images

images[0].save(output_file)
