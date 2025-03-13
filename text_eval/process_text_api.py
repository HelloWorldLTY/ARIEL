import os
import torch
os.environ['HF_HOME'] = '/home/tl688/scratch/'
os.environ['HF_TOKEN'] = ''
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset_name = 'pubmed_summary'

import pandas as pd    
file_path = f'../data/{dataset_name}_test_filter.json'
jsonObj = pd.read_json(path_or_buf=file_path)
import base64
import requests
import openai

api_key = ""

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def gpt4_chat(input_data, index):
    input_text = "Please summarize the following text: " + input_data
    payload = {
      "model": "gpt-4-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": input_text[0:127_800] #considering the maximal length
            },
          ]
        }
      ],
      "seed": 2024
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output_file = response.json()['choices']
        outdata = output_file[0]['message']['content']
    except:
        outdata = 'meet error, need retrain' + '_' + str(index)
    return outdata

model_out_data = []

for index in range(len(jsonObj)):
    print(index)
    output_data = jsonObj.loc[index].values[0]['target']
    model_output = gpt4_chat(jsonObj.loc[index].values[0]['input'], index)
    outset = {"idx":index, "output":model_output}
    model_out_data.append(outset)

import json
with open(f'/gpfs/radev/project/ying_rex/tl688/llm_output/gpt4/output_{dataset_name}.json', 'w') as f:
    json.dump(model_out_data, f)
