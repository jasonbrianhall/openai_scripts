#!/usr/bin/env python

import sys
from PIL import Image

if len(sys.argv) > 1:

  # Libraries
  from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
  import torch
  
  # Set device
  if len(sys.argv)>2:
    try:
      data=int(sys.argv[2])
    except:
      print("Invalid integer")
      data="cpu"
      pass
    device = torch.device('cuda:' + str(data))
  else:
    device="cpu"

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
  pred_prompts = predict_prompts([img], max_length=16)
  print("Predicted Text: " + pred_prompts[0])

else:
  print("Filename is missing")
