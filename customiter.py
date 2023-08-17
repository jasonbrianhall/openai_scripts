import torch
from diffusers import StableDiffusionPipeline

# Prompt user for input
prompt = input("Enter the text prompt: ")

# Prompt user for output file name
output_file = input("Enter the output file name: ")

generator = torch.Generator("cpu").manual_seed(1024)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
#image = pipe(prompt, guidance_scale=7.5, num_inference_steps=100, generator=generator).images[0]

image = pipe(prompt, guidance_scale=7.5, num_inference_steps=2, generator=generator)

counter=0
for x in image:
    print(image.images[counter])
    counter+=1

