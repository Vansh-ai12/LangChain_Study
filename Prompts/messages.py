from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain"),

]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

