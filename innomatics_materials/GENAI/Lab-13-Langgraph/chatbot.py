from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver 
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis 
from langchain_core.messages import HumanMessage
import uuid
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate
import sqlite3
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))


#state

class ChatState(MessagesState):
    messages : Annotated[list, add_messages] #Variable name should be "messages"

#initialize the node
def chat_node(state:ChatState):
    question = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will answer the user's question in a friendly and helpful manner."),
        ("human", "{messages}"),
    ])
    chain = prompt | llm
    
    response = chain.invoke({"messages": question})
    return {"messages": response}





#Graph

#initialize the graph

workflow = StateGraph(ChatState)


#add the node

workflow.add_node("chat_node", chat_node)

#Connect the nodes

workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)


#Compile the grph
# checkpointer = InMemorySaver()
# conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn)
client = Redis(
        host="redis-14921.crce220.us-east-1-4.ec2.cloud.redislabs.com",  # or your Redis Enterprise endpoint
        port=14921 ,  # or 10000 for Azure Managed Redis / Azure Enterprise with TLS
        password=os.getenv("REDIS_PASSWORD"),
       
        decode_responses=False  # RedisSaver expects bytes
    )
checkpointer = RedisSaver(redis_client=client)
checkpointer.setup()

workflow = workflow.compile(checkpointer = checkpointer)

workflow.get_graph().draw_mermaid_png(output_file_path="chatbot.png")

config = {"configurable": {"thread_id": "456"}}# thread_id is the session id for memory

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    user_input = {"messages": [HumanMessage(content=user_input)]}
    result = workflow.invoke(user_input, config)
    
    print("Bot: ",result["messages"][-1].content)