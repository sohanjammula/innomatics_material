from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model


#langchain_community -- Third party intergation -- 
from langchain_community.chat_message_histories import SQLChatMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))


def get_sql_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///conversations.db"
    )

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, who answers user questions"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

chain = prompt  | llm

# Wrap the chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_sql_history, #function to retrieve the chat history for a given session
    input_messages_key="question",
    history_messages_key="chat_history")


config = {"configurable": {"session_id": "456"}} 

while True:
    message  = input("You: ")
    if message.lower() == "exit":
        break
    response = chain_with_history.invoke(
        {"question": message},
        config=config
    )
    print("Assistant: ", response.content)