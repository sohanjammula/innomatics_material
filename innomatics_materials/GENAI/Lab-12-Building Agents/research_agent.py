from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langchain.agents import create_agent

from langgraph.checkpoint.sqlite import SqliteSaver
import os
import requests
import sqlite3
import time
from datetime import datetime
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
from langchain_exa import ExaSearchResults
load_dotenv()
import arxiv
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun


import wikipedia.wikipedia as wikipedia_client
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Wikimedia is more likely to rate-limit/block the unmaintained `wikipedia`
# package's plain HTTP request and generic User-Agent; use https + a
# descriptive UA per https://meta.wikimedia.org/wiki/User-Agent_policy.
# Note: `wikipedia/__init__.py` does `from .wikipedia import *`, which copies
# these names into the package namespace at import time — patching the
# top-level `wikipedia` module would silently no-op, since `_wiki_request`
# reads the globals of this `wikipedia.wikipedia` submodule directly.
wikipedia_client.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia_client.USER_AGENT = (
    "research_agent.py/1.0 (educational lab; contact: n/a) wikipedia-python"
)


from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

llm = init_chat_model("gpt-5-mini", 
                      model_provider="openai",
                      )

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

tools = [search_tool, arxiv_tool, pdf_report_generator, get_current_time, wiki_tool]

prompt = '''

You are an expert Research Agent. Conduct comprehensive, evidence-based research using reliable sources. Verify facts, compare relevant perspectives, identify key insights, and present findings objectively. Generate a well-structured report with clear headings, an executive summary, detailed analysis, key findings, recommendations, references, and a concise conclusion.



'''
#persistant memory
conn = sqlite3.connect("agent_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = create_agent(model = llm, tools=tools, system_prompt = prompt, checkpointer =checkpointer)

agent.get_graph().draw_mermaid_png(output_file_path="agent_react.png")

config = {"configurable": {"thread_id": "1234"}}

while True:
    user_input = input("You: ")
    inputs = {"messages": [HumanMessage(content=user_input)]}
    # result = agent.invoke(inputs, config)
    # print("Agent: ", result["messages"][-1].content)

    for event in agent.stream(inputs, config, stream_mode="values"):
        # Get the last message from the current event
        last_message = event["messages"][-1]
        
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
    
    print("\n" + "="*80 + "\n")