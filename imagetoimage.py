#!/usr/bin/env

import torch
from PIL import Image  
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda:0" 

model_id_or_path = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)

init_image_file = input("Enter sketch image file: ")
init_image = Image.open(init_image_file).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = input("Enter prompt: ")

output_file = input("Enter output file name: ")

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save(output_file)
