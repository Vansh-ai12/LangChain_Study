from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

import os
from dotenv import load_dotenv      
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
load_dotenv()

os.environ['HF_HOME'] = 'C:/Projects/huggingface_model'


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task='text-generation',
    
)

model = ChatHuggingFace(llm = llm)

chatHistory = [
    SystemMessage(content='You are a helpfull AI assistant')
]

while True:
    user_input = input('You: ')
    chatHistory.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chatHistory)