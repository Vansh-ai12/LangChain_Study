from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions = 32) 

documents = [
    "Delhi is the Capital of India.",
    "Mumbai is the Capital of Maharashtra.",
    "Kolkata is the Capital of West Bengal.",
]

result = embedding.embed_documents(documents)

print(str(result))