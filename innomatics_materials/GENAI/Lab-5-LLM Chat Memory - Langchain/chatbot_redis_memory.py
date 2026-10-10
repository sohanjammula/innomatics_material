from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
import os

load_dotenv()


llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

llm_summarize = init_chat_model("openai/gpt-oss-20b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

REDIS_URL = os.environ.get("REDIS_URL")
REDIS_KEY_PREFIX = f"{input("Enter User Id:")}:"

MAX_MESSAGES_BEFORE_SUMMARY = 8

def get_redis_history(session_id: str) :
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        key_prefix=REDIS_KEY_PREFIX,
    )
def get_chain_with_history():
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant, who answers user questions"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
            chain,
            get_redis_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    return chain_with_history


def summarize_conversation(session_id: str, summarize_mandatory: bool):
    history = get_redis_history(session_id)
    messages = history.messages
    if len(messages) <= MAX_MESSAGES_BEFORE_SUMMARY and not summarize_mandatory:
        return
    #no of messages 9
    older_messages = messages[:-2] # top 7 messages
    recent_messages = messages[-2:] #recent 2 messages
    conversation_text = "\n".join(
        f"{message.type}: {message.content}" for message in older_messages
    )
    print(f"Conversation to summarize:\n{conversation_text}")

    summary_prompt = (
        "Summarize the following conversation in a short paragraph, "
        "keeping any important facts about the user (name, preferences, etc.).\n\n"
        f"Conversation:\n{conversation_text}"
    )
    print("Summarizing conversation...")
    summary = llm.invoke(summary_prompt).content
    print(f"Summary: {summary}")
    history.clear()
    history.add_message(SystemMessage(content=f"Summary of earlier conversation: {summary}"))
    for message in recent_messages:
        history.add_message(message)


if __name__ == "__main__":
    chain_with_history = get_chain_with_history()
    config = {"configurable": {"session_id": "100"}}
    while True:
        summarize_mandatory = False
        message = input("You: ")
        if message.lower() == "exit":
            break
        if message.lower() == "summarize":
            summarize_mandatory = True
        summarize_conversation(config["configurable"]["session_id"], summarize_mandatory)
        response = chain_with_history.invoke({"input": message}, config=config)
        print("Bot: ", response.content)