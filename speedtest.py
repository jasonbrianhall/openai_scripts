#!/usr/bin/env python

import torch
from diffusers import StableDiffusionPipeline  
from multiprocessing.pool import ThreadPool
import time

def generate_image(device, prompt, filename):

  model_id = "stabilityai/stable-diffusion-2-1"

  start = time.time()
  
  pipe = StableDiffusionPipeline.from_pretrained(model_id)
  pipe.to(device)

  image = pipe(prompt).images[0]

  end = time.time()
  elapsed = end - start

  image.save(f"{filename}-{elapsed:.2f}.png")

gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
cpu = [torch.device("cpu")]

devices = gpus + cpu
prompts = ["astronaut on mars"] * len(devices)

filenames = [f"astronaut-gpu{i}" for i in range(len(gpus))]
filenames.append("astronaut-cpu")

with ThreadPool(len(devices)) as pool:
  pool.starmap(generate_image, zip(devices, prompts, filenames))
