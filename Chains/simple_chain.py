from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct", 
    task='text-generation',
    pipeline_kwargs={
        "return_full_text": False,
        "max_new_tokens": 256,
    },
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'BlackBuck'})

print(result)


