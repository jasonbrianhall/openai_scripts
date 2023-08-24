import numpy as np
import imageio
import os
import fnmatch

pattern=input("Pattern for files: ")
outputfile=input("outputfilename: ")
framesperimage=int(input("frames per image: "))

image_files = [f for f in os.listdir('.') if fnmatch.fnmatch(f, pattern)]

num_images = len(image_files)

# Load images
images = [imageio.imread(f) for f in image_files]

# Find max width and height
widths = [im.shape[0] for im in images]  
heights = [im.shape[1] for im in images]
max_width = max(widths)
max_height = max(heights)

# Pad images
padded_images = []
for im in images:
  padded = np.full((max_width, max_height, im.shape[2]), 0, dtype=np.uint8)  
  padded[:im.shape[0], :im.shape[1], :] = im
  padded_images.append(padded)

num_frames = framesperimage * num_images
frames = []

for i in np.linspace(0, 1, num=num_frames):

  # Calculate current image
  image_index = int(i * num_images)
  weight = (i % (1/num_images)) * num_images

  # Interpolate
  if image_index < num_images-1:
    current_image = padded_images[image_index]  
    next_image = padded_images[image_index+1]
    new_image = (1 - weight) * current_image + weight * next_image

  else:
    new_image = padded_images[-1]
  
  frames.append(new_image)

imageio.mimsave(outputfile, frames, fps=30)
