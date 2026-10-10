from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder #placeholder for past messages in the prompt
import os
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

_sessions = {}

#return the past messages
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, who answers user questions"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

def debug_prompt(prompt_value):
    print("\n" + "=" * 80)
    print("FINAL PROMPT")
    print("=" * 80)
    print(prompt_value.to_string())          # Nice readable format
    print("=" * 80 + "\n")
    return prompt_value

chain = prompt  | llm

# Wrap the chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history, #function to retrieve the chat history for a given session
    input_messages_key="question",
    history_messages_key="chat_history")


config = {"configurable": {"session_id": "123"}} 

while True:
    print(_sessions.keys())
    message  = input("You: ")
    if message.lower() == "exit":
        break

    if message.lower() == "new_session":
        config["configurable"]["session_id"] = input("Enter new session id: ")
        
        continue
    response = chain_with_history.invoke(
        {"question": message},
        config=config
    )
    print("Assistant: ", response.content)