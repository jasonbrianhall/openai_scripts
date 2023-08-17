#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline
import os

# Prompt user for input
prompt = input("Enter the text prompt: ")

# Prompt user for output file name
output_file = input("Enter the output file name: ")

iterations = input("iterations: ")

try:
  iterations=int(iterations)
except:
  iterations=100

print("Iterations is ", iterations)

devices=[]
for i in range(torch.cuda.device_count()):
  devices.append(f"cuda:{i}")


# Print device list  
counter=0
for i, d in enumerate(devices):
  gpu = torch.cuda.get_device_properties(i)
  print(i, "-", gpu.name)
  counter+=1
  
print(counter, "- CPU")


# Get user selection
selected = int(input("Select device: "))

if not selected==counter:
  torch.backends.cuda.matmul.allow_tf32 = True
  device = devices[selected]
  #pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype=torch.bfloat16)
  pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None)
  #pipe.enable_attention_slicing()
  pipe.enable_sequential_cpu_offload(gpu_id=int(device.split(":")[1]))
  #pipe.to(device)

else:
  pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None)
  pipe.to("cpu")


image = pipe(prompt, num_inference_steps=iterations).images[0]

image.save(output_file)
