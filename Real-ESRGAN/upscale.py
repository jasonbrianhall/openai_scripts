import torch
from PIL import Image 
from tqdm import tqdm
from RealESRGAN import RealESRGAN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model 
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth')

# Load image
img = Image.open('image.png').convert('RGB')

# Upscale image
with torch.no_grad():
  pbar = tqdm(total=2, desc='Upscaling')
  sr_img = None
  for i in range(2):   
    pbar.update(1)
    sr_img = model.predict(img)  
  pbar.close()

# Save image
sr_img.save('upscaled.png')
