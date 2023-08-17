#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler 


model_id = "stabilityai/stable-diffusion-2-1"

print("Available GPUs:")
for i in range(torch.cuda.device_count()):
  gpu = torch.cuda.get_device_properties(i)
  print(f"- GPU {i}: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB RAM)")

# Check for GPU with 8GB+ RAM
gpu_to_use = None 
for i in range(torch.cuda.device_count()):
  if torch.cuda.get_device_properties(i).total_memory >= 8 * 1024**3:
    gpu_to_use = i
    break 

print(gpu_to_use)

if gpu_to_use is not None:
  device = f"cuda:{gpu_to_use}" 
else:
  device = "cpu"  

print(f"Using device: {device}")



# Model and pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to(device)
pipe.attention_to_cpu()
pipe.set_progress_bar_config(disable=True) # reduce memory

# Enable FP16
torch.set_default_tensor_type(torch.FloatTensor) 

# Generate 256x256 image to reduce memory 
prompt = "a painting of a fox"
image = pipe(prompt, height=256, width=256).images[0]

# Clamp before uint8 conversion to avoid issues
image = (image * 255).clamp(0, 255).round().astype("uint8")

# Save image
image.save("fox.png")
