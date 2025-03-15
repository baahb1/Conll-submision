import os
import random
import getopt,sys
import shutil
import pandas as pd

import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, GenerationConfig


#~/masters
# |
#   \images_dir
#       \images
#   \code
#       \htsawyer42_HPC.py
#

HF_DATASETS_OFFLINE=1

if __name__ == "__main__":


    #model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    #model_id = "allenai/Molmo-7B-D-0924"
    HF_TOKEN = "HF_token"
    HuggingFaceKey = "HF_KEY"
    #model = MllamaForConditionalGeneration.from_pretrained(
    #    model_id,
    #    torch_dtype=torch.float16,
    #    device_map = "auto",
    #    token=HF_TOKEN,
    #)
    
    processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
    )
    #processor = AutoProcessor.from_pretrained(model_id,token=HF_TOKEN)

    filePaths = []
    for root,dirs,files in os.walk("../images_50"):
        for file in files:
            filePaths.append(os.path.join(root,file))


    Finished_path = "./HPC_Finished.csv"
    try:
      responses = pd.read_csv(Finished_path,index_col=0)
    except EOFError:
      responses = pd.DataFrame(columns=['files','output'])
    except FileNotFoundError:
      responses = pd.DataFrame(columns=['files','output'])
    print(responses)



    path_experts = "./HPC_EXPERT.csv"
    
    Finished_Path_Experts = "./HPC_Finished_Experts.csv"
    try:
      experts = pd.read_csv(Finished_Path_Experts,index_col=0)
    except EOFError:
      experts = pd.DataFrame(columns=['files','output'])
    except FileNotFoundError:
      experts = pd.DataFrame(columns=['files','output'])






#normal Prompting
count = 50
for file in filePaths:
  if file in responses['files'].values:
    continue
  #image = Image.open(file)
  #messages = [
  #    {"role": "user", "content": [
  #        {"type": "image"},
  #        {"type": "text", "text": "Describe the main subject in this image in minimal words"}
  #    ]}
  #]


  inputs = processor.process(
  images=[Image.open(file)],
  text="Describe the main subject in this image in minimal words."
  )
    
  inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
  #input_text = processor.apply_chat_template(
  #    messages, add_generation_prompt=True,
  #)

  #inputs = processor(
  #    image,
  #    input_text,
  #    add_special_tokens=False,
  #    return_tensors="pt",
  #).to(model.device)
  
  
  with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    output = model.generate_from_batch(
      inputs,
      GenerationConfig(max_new_tokens=30, stop_strings="<|endoftext|>"),
      tokenizer=processor.tokenizer
  )
  
  generated_tokens = output[0,inputs['input_ids'].size(1):]
  generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
  
  #output = model.generate(**inputs, max_new_tokens=30)
  #decodedOutput = processor.decode(output[0])
  #decodedOutput = str(decodedOutput).replace(",","")
  generated_text = str(generated_text).replace(",","")
  print(generated_text)
  print(file)
  new_response = {'files': file, 'output': generated_text}
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
  #messages = [
  #    {"role": "user", "content": [
  #        {"type": "image"},
  #        {"type": "text", "text": "Say the word for experts in the field of whatever the primary object in the image is, then say that you are one of whatever the term is. Then acting as this expert, describe the main subject in minimal words"},
  #    ]}

  #]
  #Identify the primary object in this image. after you have identified it, say what an expert in that object would be called. pretend to be one of these experts then describe what the object is"
  #Describe the main subject in this image in minimal words
  #Say the word for experts in the field of whatever the primary object in the image is, then say that you are one of whatever the term is. Then acting as an this expert, describe the main subject in minimal words


  inputs = processor.process(
  images=[image],
  text="Say the word for experts in the field of whatever the primary object in the image is, then say that you are one of whatever the term is. Then acting as this expert, describe the main subject in minimal words."
  )
    
  inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
  
  #input_text = processor.apply_chat_template(
  #    messages, add_generation_prompt=True,
  #)

  #inputs = processor(
  #    image,
  #    input_text,
  #    add_special_tokens=False,
  #    return_tensors="pt",
  #).to(model.device)
  
  output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=30, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
  )
  
  generated_tokens = output[0,inputs['input_ids'].size(1):]
  decodedOutput = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
  
  #output = model.generate(**inputs, max_new_tokens=30)
  #decodedOutput = processor.decode(output[0])
  decodedOutput = str(decodedOutput).replace(",","")
  
  print(decodedOutput)
  print(file)
  new_response = {'files': file, 'output': decodedOutput}
  experts = experts._append(new_response, ignore_index=True)
  if(count >= 200):
    experts.to_csv(Finished_Path_Experts)
    count = 0
  count+=1

experts.to_csv(Finished_Path_Experts)

print("finished_experts")
