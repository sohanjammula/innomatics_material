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
from langchain_exa import ExaSearchResults
from datetime import datetime
from langchain_core.tools import tool
import time
load_dotenv()

import arxiv
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.messages import ToolMessage
import json

import wikipedia.wikipedia as wikipedia_client
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
wikipedia_client.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia_client.USER_AGENT = (
    "research_agent.py/1.0 (educational lab; contact: n/a) wikipedia-python"
)

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

llm = init_chat_model("claude-sonnet-5", 
                      model_provider="anthropic")


#state

exa_search = ExaSearchResults(
    exa_api_key=os.environ["EXA_API_KEY"],
    max_results=1,
)


def _fetch_results_via_client(self, query: str):
    """arxiv>=2.0 removed Search.results(); results now come from Client.results(search)."""
    if self.is_arxiv_identifier(query):
        search = self.arxiv_search(id_list=query.split(), max_results=self.top_k_results)
    else:
        search = self.arxiv_search(
            query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
        )
    return arxiv.Client().results(search)


ArxivAPIWrapper._fetch_results = _fetch_results_via_client

api_wrapper = ArxivAPIWrapper(
    top_k_results=3,               # Default is 3
    doc_content_chars_max=5000     # Cutoff limit for summary text length
)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper)


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


@tool
def search_tool(query: str) -> str:
    """
    Search for information using the ExaSearchResults tool.

    Args:
        query (str): The search query.
    Returns:
        str: Search results as a string.
    """ 
    try:
        results = exa_search._run(
            query=query,
            num_results=1,
            text_contents_options=True,
            highlights=True,
        )
        return str(results)
    except Exception as e:
        return f"Error during search: {str(e)}"



@tool
def pdf_report_generator(text: str, file_name: str):
    """
    Generates a PDF report from the provided text and saves it to the specified file name.

    Args:
        text (str): The content to be included in the PDF report.
        file_name (str): The name of the PDF file to be created. If the file name does not end with '.pdf', the extension will be added automatically.

    Returns:
        None

    The function formats the text into paragraphs, applies basic styling, and creates a PDF document titled "Research Report".
    """
    try:
        
        # Ensure file has .pdf extension
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'
        
        # Create PDF document
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_LEFT,
            leftIndent=20
        )
        
        # Add title
        title = Paragraph("Research Report", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Split text into paragraphs and add to story
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Clean up the paragraph text
                clean_para = para.strip().replace('\n', ' ')
                if clean_para:
                    story.append(Paragraph(clean_para, body_style))
                    story.append(Spacer(1, 10))
        
        # Build PDF
        doc.build(story)
        
        # Get absolute path for confirmation
        abs_path = os.path.abspath(file_name)
        
        return f"PDF report successfully saved as '{file_name}' at {abs_path}"
        
    except ImportError:
        return "Error: reportlab library not installed. Please install it with: pip install reportlab"
    except Exception as e:
        return f"Error creating PDF: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current date and time in YYYY-MM-DD HH:MM:SS format
    """
    print("Getting current time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_current_time, search_tool, wiki_tool, pdf_report_generator, arxiv_tool]


llm = llm.bind_tools(tools)


class ChatState(MessagesState):
    messages : Annotated[list, add_messages] #Variable name should be "messages"

#initialize the node
def chat_node(state:ChatState):
    question = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''
        You are a helpful assistant. Answer the user's questions clearly, accurately, and in a friendly manner. Use the available tools whenever they are relevant to the user's request.

        Available tools:

        1. **get_current_time** — Get the current date and time.
        2. **search_tool** — Search the web for relevant information using ExaSearchResults.
        3. **wiki_tool** — Search Wikipedia for information about a topic.
        4. **pdf_report_generator** — Generate a PDF report from provided text.
        5. **arxiv_tool** — Search arXiv for relevant academic papers.

        
        
        '''),
        ("human", "{messages}"),
    ])
    chain = prompt | llm
    
    response = chain.invoke({"messages": question})
    return {"messages": response}


tools_by_name = {tool.name: tool for tool in tools}
# Define our tool node
def tool_node(state: ChatState):
    outputs = []
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        try:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])#call the tool
        except Exception as e:
            print(f"Error calling tool {tool_call['name']}: {e}")
            tool_result = f"Error calling tool {tool_call['name']}: {e}"
       
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}
#Graph

#initialize the graph

workflow = StateGraph(ChatState)


#add the node

workflow.add_node("chat_node", chat_node)
workflow.add_node("tools", tool_node)

#Connect the nodes
def should_call_tools(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "tools"
    
workflow.add_edge(START, "chat_node")

workflow.add_conditional_edges("chat_node", 
                               should_call_tools,
                               {"tools": "tools", "end": END})

workflow.add_edge("tools", "chat_node")



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

workflow.get_graph().draw_mermaid_png(output_file_path="chatbot_agent.png")

config = {"configurable": {"thread_id": "3456"}}# thread_id is the session id for memory

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    user_input = {"messages": [HumanMessage(content=user_input)]}
    # result = workflow.invoke(user_input, config)
    for output in workflow.stream(user_input, config=config):
        for key, value in output.items():
            last_message = value["messages"]
            print(last_message) 
            # Handle different message types
            if hasattr(last_message, 'type'):
                if last_message.type == "ai":
                    # Agent's reasoning or response
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        # Agent is calling a tool
                        print("\n🤖 Agent Decision: Calling tool(s)")
                        for tool_call in last_message.tool_calls:
                            print(f"   📌 Tool: {tool_call['name']}")
                            print(f"   📝 Args: {tool_call['args']}")
                    elif last_message.content:
                        # Agent's final response
                        print(f"\n🤖 Bot: {last_message.content}")
                
                elif last_message.type == "tool": # this is the tool call result
                    # Tool execution result
                    print(f"\n🔧 Tool Output ({last_message.name}):")
                    print(f"   ✅ Result: {last_message.content}")
    # print("Bot:", value["messages"][-1].content)