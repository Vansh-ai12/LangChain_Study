from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report on {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text'],

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

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployment in India'})

print(result)


chain.get_graph().print_ascii()