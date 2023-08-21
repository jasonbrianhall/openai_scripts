#!/usr/bin/env python

import sys
from PIL import Image
import torch
import random
import warnings

warnings.filterwarnings("ignore")

if len(sys.argv) > 1:

  # Get capable GPU if available
  gpus = torch.cuda.device_count()
  available_gpus = []
  for i in range(gpus):
      cuda_cap = torch.cuda.get_device_capability(i)
      if cuda_cap[0] >= 3
          if cupa_cap[0]==3 and cuda_cap[1] >= 7:
              available_gpus.append(i)
          elif cupa_cap[0]>3: 
              available_gpus.append(i)


  if len(available_gpus) > 0:
      device = torch.device(f'cuda:{random.choice(available_gpus)}')
  else:
      device = torch.device('cpu')  

  # Libraries
  from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

  # Load model
  model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
  model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

  # Predict function
  def predict_prompts(list_images, max_length=16):
    model.eval()
    pixel_values = feature_extractor(images=list_images, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds
  
  # Get image and predict
  img = Image.open(sys.argv[1]).convert('RGB')
  pred_prompts = predict_prompts([img], max_length=500)
  print("Filename: " + sys.argv[1] + "\nPredicted Text: " + pred_prompts[0])

else:
  print("Filename missing")
