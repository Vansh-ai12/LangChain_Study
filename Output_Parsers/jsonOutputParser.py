from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct", 
    task='text-generation',
    pipeline_kwargs={
        "return_full_text": False,
        "max_new_tokens": 256,
    },
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name , age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})

print(result)