import os
os.environ['HF_HOME'] = '/home/tl688/scratch/'
os.environ['HF_TOKEN'] = '' #fill huggingface token
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset_name = 'pubmed_summary'
import pandas as pd    
# file_path = f'../data/pubmed_summary_test_filter_filtertextNone_needskipnoneout.json'
file_path = f'../data/pubmed_summary_test_filter_filtertextNone_cleanabstract.json'
jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
jsonObj = jsonObj.T

model_out_data = []

device_map = 'auto'
model_name="THUDM/glm-4-9b-chat-1m"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map, trust_remote_code=True)

def summarize_text(text, model_name="meta-llama/Meta-Llama-3-8B", max_new_tokens=100, device_map='auto'): #max new tokens should be the same as abstract length.
    # Encode the text and generate summary
    inputs = tokenizer(text, return_tensors="pt", truncation=False, max_length=1000000 - max_new_tokens)
    attention_mask = inputs["attention_mask"]
    summary_ids = model.generate(inputs.input_ids.to(model.device), attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary[len(text):]

for index in range(len(jsonObj)):
    input_data = "Please summarize the following text: " + jsonObj.loc[index][0]['input']
    output_data = jsonObj.loc[index][0]['target']
    model_output = summarize_text(text=input_data, model_name=model_name, max_new_tokens = len(output_data))
    outset = {"idx":index, "output":model_output}
    model_out_data.append(outset)


# import os
# os.makedirs('/gpfs/radev/scratch/ying_rex/tl688/llm_output/chatglm4_1m/')
import json
with open('/gpfs/radev/project/ying_rex/tl688/llm_output/chatglm4_1m/pubmed_summary_chatglmunlimit_update.json', 'w') as f:
    json.dump(model_out_data, f)
