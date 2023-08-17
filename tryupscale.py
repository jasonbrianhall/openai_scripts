from PIL import Image
import torch
import stable_diffusion

model_id = "stabilityai/stable-diffusion-2-1"
model = stable_diffusion.load_model(model_id)

image_path = "dwarves.png"
image = Image.open(image_path)

width, height = image.size  
new_width = width * 2
new_height = height * 2

prompt = f"Enlarge this {width}x{height} pixel image to {new_width}x{new_height} pixels:"

batch = stable_diffusion.Imagine(prompt=prompt) 
scaled_image = model.imagine(batch)[0]

scaled_image.save("scaled_image.jpg")
