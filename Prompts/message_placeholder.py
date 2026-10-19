# Message placeholder gives the past conversations context
#to retrieve and store chat history


from langchain_core.prompts import ChatPromptTemplate , MessagePlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
# chat template

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagePlaceholder(variable_name='chat_history')
    ('human','{query}')
])


chatHistory=[]

# load chat history
with open('chat_history.txt') as f:
    chatHistory.append(f.readlines())

print(chatHistory)

#create prompt

prompt = chat_template.invoke({'chat_history':chatHistory,'query':'What is the status of my order?'})

print(prompt)















