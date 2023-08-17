#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os


# Prompt user for input
prompt = input("Enter the text prompt: ")

# Prompt user for output file name
output_file = input("Enter the output file name: ")

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Check if CUDA is available and meets the minimum required level
cuda_available = torch.cuda.is_available()
#cuda_min_version = torch.version.cuda.split('.')[0]
#cuda_required_version = '11'  # Set your minimum required CUDA version here

torch.set_num_threads(16)
torch.set_default_tensor_type(torch.cuda.HalfTensor)

if cuda_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


image = pipe(prompt).images[0]

image.save(output_file)

