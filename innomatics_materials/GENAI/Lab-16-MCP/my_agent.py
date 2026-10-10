from langchain.agents import create_agent

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
load_dotenv() 


GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
MCP_SERVERS = {
    "my_tools": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
    "context7": {
        "url": "https://mcp.context7.com/mcp",
        "transport": "streamable_http",
    },
    "github": {
        "url": "https://api.githubcopilot.com/mcp/",
        "transport": "streamable_http",
        "headers": {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
        },
    }
}

client = MultiServerMCPClient(MCP_SERVERS)

tools_mcp= asyncio.run(client.get_tools())

search_tool = DuckDuckGoSearchRun()

_wiki_api_wrapper = WikipediaAPIWrapper()

@tool
def wiki_tool(query: str) -> str:
    """
    Search Wikipedia for information on a topic.

    Args:
        query (str): The search query.
    Returns:
        str: Wikipedia summary results as a string.
    """
    
    for attempt in range(2):
        try:
            return _wiki_api_wrapper.run(query)
        except Exception as e:
            if attempt == 0:
                time.sleep(5)
                continue
            return f"Error during Wikipedia search: {str(e)}"
llm = init_chat_model("gpt-5-mini", 
                      model_provider="openai",
                      )

prompt = asyncio.run(client.get_prompt("my_tools", "helpful_assistant"))

checkpointer = InMemorySaver()
agent = create_agent(
    model=llm,
    tools=[search_tool, wiki_tool] + tools_mcp,
    system_prompt=prompt[0].content,
    checkpointer=checkpointer)


config = {"configurable": {"thread_id": "1"}}


async def run_agent():
    while True:
        user_input = input("You: ")
        inputs = {"messages": [HumanMessage(content=user_input)]}

        async for event in agent.astream(inputs, config, stream_mode="values"):
            last_message = event["messages"][-1]

            if hasattr(last_message, 'type'):
                if last_message.type == "ai":
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        print("\n🤖 Agent Decision: Calling tool(s)")
                        for tool_call in last_message.tool_calls:
                            print(f"   📌 Tool: {tool_call['name']}")
                            print(f"   📝 Args: {tool_call['args']}")
                    elif last_message.content:
                        print(f"\n🤖 Bot: {last_message.content}")

                elif last_message.type == "tool":
                    print(f"\n🔧 Tool Output ({last_message.name}):")
                    print(f"   ✅ Result: {last_message.content}")

        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_agent())
