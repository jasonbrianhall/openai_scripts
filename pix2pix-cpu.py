import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to("cpu")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

input_file = input("Enter image filename: ")
prompt = input("Enter prompt: ")
output_file = input("Enter output filename: ")

image = PIL.Image.open(input_file)
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")

images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0].save(output_file)
