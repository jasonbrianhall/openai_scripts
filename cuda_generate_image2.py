#!/bin/bash

import torch
from diffusers import StableDiffusionPipeline
import deepspeed 

# Model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Get underlying PyTorch model
#model = pipe.model
#model = pipe.decoder
model = pipe.unet

# DeepSpeed config
ds_config = {
  "train_batch_size": 1,
  "fp16": {
     "enabled": True
  },
  "zero_optimization": {
     "stage": 1,
     "offload_param": {
       "device": "cpu"
     }
  }
}

# Initialize DeepSpeed engine
model_engine, _, _, _ = deepspeed.initialize(args=ds_config,
                                             model=pipe,
                                             model_parameters=model.parameters())
                                             
# DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) 

# Training loop
for epoch in range(epochs):

  for i, batch in enumerate(dataloader):

    # Forward and backward pass
    loss = model_engine(batch)
    model_engine.backward(loss)  
    model_engine.step()
    
# Generation
prompt = "a painting of a fox"
with torch.no_grad():
  images = model_engine(prompt).images
  
# Save image  
image = images[0]
image.save("fox.png")
