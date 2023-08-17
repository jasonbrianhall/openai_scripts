import torch
import requests 
import os

# Weights details
url = "https://github.com/Stability-AI/stablediffusion/releases/download/v2-1/stable-diffusion-2-1-fp16.ckpt"
local_path = './model.ckpt'

# Check if weights file exists
if not os.path.exists(local_path):

  # Download weights
  print("Downloading model weights...")
  response = requests.get(url)
  
  with open(local_path, "wb") as f:
    f.write(response.content)
    
  print("Weights downloaded to", local_path)

# Load model
pipe = torch.load(local_path)
pipe = pipe.to("cpu")

print("Model loaded.")

# Sample prompts 
prompts = ["a painting of a cat", "a photo of the Eiffel tower"]
output_files = ["cat.png", "eiffel.png"] 

# Generate images
for prompt, outfile in zip(prompts, output_files):
  image = pipe(prompt).images[0]
  torch.save(image, outfile)

print("Images generated successfully!")
