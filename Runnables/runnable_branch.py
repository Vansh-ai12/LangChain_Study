from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

from langchain_core.runnables import RunnableSequence ,RunnablePassthrough ,RunnableLambda , RunnableParallel , RunnableBranch

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = ' Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1 , model , parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(report_chain, branch_chain)

result = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(result)