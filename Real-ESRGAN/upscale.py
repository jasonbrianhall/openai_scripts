import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

filename=input("Filename to upscale:  ")

image = Image.open(filename).convert('RGB')

sr_image = model.predict(image)

filename_split=filename.split(".")[0]

sr_image.save(filename_split + "_upscaled.png")
