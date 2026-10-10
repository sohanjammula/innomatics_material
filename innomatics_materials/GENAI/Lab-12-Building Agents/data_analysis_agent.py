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
from typing import Tuple
from langgraph.checkpoint.memory import InMemorySaver
from langchain_experimental.tools import PythonAstREPLTool  #Python code execution tool
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER

load_dotenv()



os.environ["LANGSMITH_PROJECT"] = "data-analysis-agent"
llm = init_chat_model(model="gpt-5-mini", model_provider="openai")

@tool
def execute_code(code: str, csv_path: str) -> Tuple[bool, str, float]:
        """Execute code with timeout and memory monitoring using PythonAstREPLTool"""
        # print(f"Executing code: {code} with csv path: {csv_path}")
        
        try:
            # Initialize PythonAstREPLTool
            python_repl = PythonAstREPLTool()
            
            # Prepare the complete code that sets up the environment and executes user code
            complete_code = f"""
# Setup environment
import pandas as pd


# Execute the user's pandas code
{code}
"""
            
            # Execute with timeout using asyncio
            result = python_repl.run(complete_code)
            return result
        except Exception as e:
            
            return f"Execution error: {str(e)}"

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
tools = [execute_code, pdf_report_generator]


data_analysis_prompt = """
You are a skilled Data Analyst. 
You have access to a tool called execute_code that allows you to run code to analyze and extract insights.
Use this tool to explore the data, perform calculations, and answer the user's questions accurately and concisely based on the Data.

Focus your pandas code analysis strictly on what is required to answer the user's question, rather than exploring the entire dataset.

You can find the dataset in the data/ folder.

"""

checkpointer = InMemorySaver()

data_analysis_agent = create_agent(model=llm, tools=tools, system_prompt=data_analysis_prompt, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "123"}}


while True:
    user_input = input("You: ")
    user_question = f'''
    {user_input}

    '''
    # result = data_analysis_agent.invoke({"messages": [HumanMessage(content=user_question)]}, config=config)
    # print("Agent: ", result["messages"][-1].content)

    for event in data_analysis_agent.stream({"messages": [HumanMessage(content=user_question)]}, config, stream_mode="values"):
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
