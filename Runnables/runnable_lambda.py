from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence ,RunnablePassthrough ,RunnableLambda , RunnableParallel

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


model = ChatOpenAI()

parser = StrOutputParser()


joke_chain = RunnableSequence(prompt , model ,parser)

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'word_count':RunnableLambda(lambda x: len(x.split()))
    }
)

final_chain = RunnableSequence(joke_chain , parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result['joke'])
print(result['word_count'])
