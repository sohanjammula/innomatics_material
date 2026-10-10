from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

client = MultiServerMCPClient({
    "my_server": {
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
})


tools = asyncio.run(client.get_tools(server_name="github"))

print("Tools available from MCP server:")
print(tools)

for tool in tools:
    print("Tool Name: ",tool.name)
    print("Tool Description: ",tool.description)
    print("---")

# prompt = asyncio.run(client.get_prompt("my_server","helpful_assistant"))
# print(prompt)


# resources = asyncio.run(client.get_resources("my_server"))
# print(resources)
# for resource in resources:
#     # print(resource.name)
#     print(str(resource.metadata["uri"]).split("://")[1])
#     resource_name = str(resource.metadata["uri"]).split("://")[1]
#     print(resource.mimetype)
#     if resource_name == "test_doc.txt/":
#         print(resource.data)
#     # print(resource.uri)
#     # print(resource.data)