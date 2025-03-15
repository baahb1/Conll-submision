import os
import random
import getopt,sys
import pickle
import shutil

import requests
import torch
import torch_directml
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from accelerate import infer_auto_device_map


HF_DATASETS_OFFLINE=1

def Respond_to_image(file_path):
    print("hello")

def ShrinkDataset(threshold,path="../images/train/"):
    print("hello")
    folders = os.listdir(path)
    NewFiles = []
    Categories = []
    for i in folders:
        files = os.listdir(path + i)
        for f in files:  
            if random.random() <=threshold:
                NewFiles.append(path + i +"/"+ f)
                Categories.append(i.split("_")[1])
    return NewFiles,Categories

def shrinkDatasetByNumber(number,path="D:/ecoset/train/"):
    folders = os.listdir(path)
    NewFiles = []
    Categories = []
    for i in folders:
        files = os.listdir(path + i)
        count = 0  
        while count < number:
            randfile = random.choice(files)
            files.remove(randfile)
            count +=1
            NewFiles.append(path+ i + "/" + randfile)
            Categories.append(i.split("_")[1])
    return NewFiles,Categories

def newFolders(newFiles:list,Categories):
    print("transfer")
    try:
        os.mkdir("../images_50")
    except:
        pass
    for f in Categories:
        try:
            os.mkdir("../images_50/"+ f)
        except:
            pass
    for f in range(len(newFiles)):
        print(newFiles[f])
        print("../images_50/" + Categories[f] + "/" + newFiles[f].split("/")[4])
        shutil.copyfile(newFiles[f],"../images_50/" + Categories[f] + "/" + newFiles[f].split("/")[4])
        #print(str("../images2/" + Categories[f] + "/" + newFiles[f].split("/")[4]))
        #print(newFiles[f])







if __name__ == "__main__":
    threshold = 1
    inputFile = None
    shrink = False
    outputFile = "files.txt"
    transfer = False
    #options handling, taken largely from geeks for geeks
    argList = sys.argv[1:]
    options = "opi"
    long_options = ["threshold=","output=","path=","shrink=","inputFile=","transfer="]

    

    try:
        arguments, values = getopt.getopt(argList,options,long_options)

        for currentArgument, CurrentValue in arguments:
            if currentArgument in ("--threshold"):
                threshold = float(CurrentValue)
            if currentArgument in ("-o","--output"):
                outputFile = str(CurrentValue)
            if currentArgument in ("--shrink"):
                if(str(CurrentValue).strip().upper() in ["True","T"]):
                    shrink = True
            if currentArgument in ("-i","--input"):
                inputFile = str(CurrentValue)
            if currentArgument in ("--transfer"):
                if(str(CurrentValue).strip().upper() in ["True","T"]):
                    transfer = True
            
    except getopt.error as err:
        print("opterror")


    if shrink == True and inputFile == None:
        #NewFiles,categories = ShrinkDataset(threshold)
        NewFiles,categories = shrinkDatasetByNumber(50)
        print("shrinking")
        
        with open('files.pkl', 'wb') as file: 
            pickle.dump(NewFiles, file) 

        with open('categories.pkl', 'wb') as file: 
            pickle.dump(categories, file) 
                
    if shrink == False or inputFile != None:
        with open('categories.pkl', 'rb') as file: 
            categories = pickle.load(file) 
        with open('files.pkl', 'rb') as file: 
            NewFiles = pickle.load(file)

    if(transfer == True):
        newFolders(NewFiles,categories)

    print(NewFiles[0])
    print(categories[0])
    print(shrink)
    print(threshold)

    #HUGGING FACE VLM 3.2 CODE BELOW ----------------------------------------------------------------------------- only runs on cpu for my amd gpu
    
    #device_map_gpu = {'vision_model': "privateuseone", 'language_model.model.embed_tokens': 'cpu', 'language_model.model.layers.0': 'cpu', 'language_model.model.layers.1': 'cpu', 'language_model.model.layers.2': 'cpu', 'language_model.model.layers.3': 'cpu', 'language_model.model.layers.4': 'cpu', 'language_model.model.layers.5': 'cpu', 'language_model.model.layers.6': 'cpu', 'language_model.model.layers.7': 'cpu', 'language_model.model.layers.8': 'cpu', 'language_model.model.layers.9': 'cpu', 'language_model.model.layers.10': 'cpu', 'language_model.model.layers.11': 'cpu', 'language_model.model.layers.12': 'cpu', 'language_model.model.layers.13': 'cpu', 'language_model.model.layers.14': 'cpu', 'language_model.model.layers.15': 'cpu', 'language_model.model.layers.16': 'cpu', 'language_model.model.layers.17': 'cpu', 'language_model.model.layers.18': 'cpu', 'language_model.model.layers.19': 'cpu', 'language_model.model.layers.20': 'cpu', 'language_model.model.layers.21': 'cpu', 'language_model.model.layers.22': 'cpu', 'language_model.model.layers.23': 'cpu', 'language_model.model.layers.24': 'cpu', 'language_model.model.layers.25': 'cpu', 'language_model.model.layers.26': 'cpu', 'language_model.model.layers.27': 'cpu', 'language_model.model.layers.28': 'cpu', 'language_model.model.layers.29': 'cpu', 'language_model.model.layers.30': 'cpu', 'language_model.model.layers.31': 'cpu', 'language_model.model.layers.32': 'cpu', 'language_model.model.layers.33': 'cpu', 'language_model.model.layers.34': 'cpu', 'language_model.model.layers.35': 'cpu', 'language_model.model.layers.36': 'cpu', 'language_model.model.layers.37': 'disk', 'language_model.model.layers.38': 'disk', 'language_model.model.layers.39': 'disk', 'language_model.model.norm': 'disk', 'language_model.model.rotary_emb': 'disk', 'language_model.lm_head': 'disk', 'multi_modal_projector': 'disk'}
    '''
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map = "auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "can you please discribe this image"}
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

    

    #output = model.generate(**inputs, max_new_tokens=30)
    #print(processor.decode(output[0]))
    '''