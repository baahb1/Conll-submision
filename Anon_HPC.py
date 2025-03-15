import os
import random
import getopt,sys
import pickle
import shutil
import pandas as pd

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


#~/masters
# |
#   \images_dir
#       \images
#   \code
#       \htsawyer42_HPC.py
#

HF_DATASETS_OFFLINE=1

if __name__ == "__main__":


    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    HuggingFaceKey = "your_key"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map = "auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(model_id,token=HF_TOKEN)

    filePaths = []
    for root,dirs,files in os.walk("../images_50"):
        for file in files:
            filePaths.append(os.path.join(root,file))


    Finished_path = "./HPC_Finished.csv"
    try:
        responses = pd.read_csv(Finished_path)
    except EOFError:
        responses = pd.DataFrame(columns=['files','output'])
    except FileNotFoundError:
        responses = pd.DataFrame(columns=['files','output'])
    print(responses)



    path_experts = "./HPC_EXPERT.csv"
    
    Finished_Path_Experts = "./HPC_Finished_Experts.csv"
    try:
      experts = pd.read_csv(Finished_Path_Experts)
    except EOFError:
      experts = pd.DataFrame(columns=['files','output'])
    except FileNotFoundError:
        responses = pd.DataFrame(columns=['files','output'])






#normal Prompting
count = 50
for file in filePaths:
  if file in responses['files'].values:
    continue
  image = Image.open(file)
  messages = [
      {"role": "user", "content": [
          {"type": "image"},
          {"type": "text", "text": "Describe the main subject in this image in minimal words"}
      ]}
  ]

  input_text = processor.apply_chat_template(
      messages, add_generation_prompt=True,
  )

  inputs = processor(
      image,
      input_text,
      add_special_tokens=False,
      return_tensors="pt",
  ).to(model.device)
  output = model.generate(**inputs, max_new_tokens=30)
  decodedOutput = processor.decode(output[0])
  decodedOutput = str(decodedOutput).replace(",","")
  print(decodedOutput)
  print(file)
  new_response = {'files': file, 'output': decodedOutput}
  responses = responses._append(new_response, ignore_index=True)
  if(count >= 50):
    responses.to_csv(Finished_path)
    count = 0
  count+=1
responses.to_csv(Finished_path)



#EXPERT PROMPTINGS
count = 50
for file in responses["files"]:
  if file in experts["files"].values:
    continue

  image = Image.open(file)
  messages = [
      {"role": "user", "content": [
          {"type": "image"},
          {"type": "text", "text": "describe the object in the image"},
      ]}

  ]
  #Identify the primary object in this image. after you have identified it, say what an expert in that object would be called. pretend to be one of these experts then describe what the object is"
  #Describe the main subject in this image in minimal words
  #Say the word for experts in the field of whatever the primary object in the image is, then say that you are one of whatever the term is. Then acting as an this expert, describe the main subject in minimal words

  input_text = processor.apply_chat_template(
      messages, add_generation_prompt=True,
  )

  inputs = processor(
      image,
      input_text,
      add_special_tokens=False,
      return_tensors="pt",
  ).to(model.device)
  output = model.generate(**inputs, max_new_tokens=30)
  decodedOutput = processor.decode(output[0])
  print(decodedOutput)
  print(file)
  new_response = {'files': file, 'output': decodedOutput}
  experts = experts._append(new_response, ignore_index=True)
  if(count >= 50):
    experts.to_csv(Finished_Path_Experts)
    count = 0
  count+=1

experts.to_csv(Finished_Path_Experts)
