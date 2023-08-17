import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

local_filename = input("Enter local filename: ")
prompt = input("Enter prompt: ")
output_filename = input("Enter output filename: ")

# Load model
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to("cpu")

# Load original image
img = Image.open(local_filename).convert("RGB")
orig_width, orig_height = img.size

# Calculate 4x dimensions 
upscaled_width = orig_width
upscaled_height = orig_height

# Resize image to 4x size
img = img.resize((upscaled_width, upscaled_height))

# Upscale image
upscaled_image = pipeline(prompt=prompt, image=img).images[0] 

# Save image
upscaled_image.save(output_filename)
