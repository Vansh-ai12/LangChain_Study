from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
url = 'https://www.amazon.in/HP-Smartchoice-Upgradeable-Office24-fb3130AX/dp/B0GVRZD89W/ref=sr_1_1?crid=R65TSUO501WJ&dib=eyJ2IjoiMSJ9.TrgkObzQU2UUNPxPa5bnntBM-qwnXQqgLudKKOJWWJ_dPbBTCO7yj-GSPoPoDJwv-rdDRBf9uyS1kO87kHX_jR3_9mv4oJhvEdukBazIk4YQJsnrRdGw5jBeO2WUJCEoAYs7f_uvKbYQp2euH-du4ElQJILAwsst9B2iD_2NlzXIS_-FqzniiKF2NcpeJDDjRBMWX1FxG16vjHpM7rmFyBAqWuoRSc09rODxy3J1aoM.qOzpvnab5ZRmbTMWgKyYr4Qdz1Z5t1mgpsI5nHxlM6s&dib_tag=se&keywords=Victus&qid=1792149474&sprefix=victu%2Caps%2C270&sr=8-1&th=1'

loader = WebBaseLoader(url)

prompt = PromptTemplate(
    template='Answer the following question {question} from the following text {text}',
    input_variables=['question','text']
)

model = ChatOpenAI()

parser = StrOutputParser()

docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'question':'What is the Price of the laptop' , 'text':docs[0].page_content})

print(result)


