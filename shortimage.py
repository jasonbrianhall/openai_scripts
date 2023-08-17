#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline

# Prompt user for input
prompt = input("Enter the text prompt: ")

# Prompt user for output file name
output_file = input("Enter the output file name: ")

generator = torch.Generator("cpu").manual_seed(1024)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=5, generator=generator).images[0]

# you can save the image with
image.save(output_file)

