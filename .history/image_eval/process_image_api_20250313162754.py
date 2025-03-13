import base64
import requests

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "./figure_dict/fig_test_1.png" # using figure 1 as an example, the other figures are the same.

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": '''Acting as a bioinformatics professional proficient in RNA-Seq data analysis, you are interpreting a figure to better appreciate transcription response of multiple myeloma cells (MMs) to bone marrow stromal cells (BMSCs). You also have extensive laboratory research experience in bone marrow tumor microenvironment of MM. In this task, expression profile of MM cell line RMPI8226 in trans-well coculture (T) with a BMSC cell line was compared to in monoculture (M). The attached figure illustrates the differential expression between the two conditions. Please do the following: 
1. Provide an overview on each panel. 
2. For panel A, what do the numbers in parentheses mean? 
3. For panel B, the positions of SOCS3 and JUNB are indicated by arrow heads. Estimate the values from the x-axis for the two genes. Are the two genes up-regulated or down-regulated? Based on your findings in the literature, why are the two genes chosen and are these results expected? 
4. For panel C, pick the top two pathways and explain the criteria. It is not indicated on the plot whether upregulated or downregulated genes were used for the pathway enrichment analysis. However, by correlating the findings to existing knowledge in the literature, which group of genes were likely used? 
5. Are there any specific suggestions to improve the data presentation of the figure? 
6. Draft a title and figure legend. Make sure to include details. 
7. Draft a paragraph to describe the findings from the figure. This is one paragraph for result part of a manuscript. 150 words or less. '''
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "seed": 2024
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())