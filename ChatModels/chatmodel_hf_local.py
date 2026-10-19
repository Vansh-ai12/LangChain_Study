from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

import os
from dotenv import load_dotenv      

load_dotenv()

os.environ['HF_HOME'] = 'C:/Projects/huggingface_model'


llm = HuggingFacePipeline.from_model_id(
    model_id="openai/gpt-oss-20b",
    task='text-generation',
    
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of India?")  

print(result.content)