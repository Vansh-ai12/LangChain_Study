from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel , Field

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

class Person(BaseModel):
    name: str = Field(description='Name of the Person')
    age: int = Field(gt=18, description = 'Age of the Person')
    city : str = Field(description='Name of the city the person belongs to ')


parser = PydanticOutputParser(
    pydantic_object=Person
)

template = PromptTemplate(
    template = "Generate the name ,age and the city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'England'})


print(result)