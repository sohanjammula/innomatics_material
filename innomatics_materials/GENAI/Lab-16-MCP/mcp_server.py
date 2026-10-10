from datetime import datetime
from starlette.applications import Starlette
from starlette.routing import Mount
from contextlib import AsyncExitStack, asynccontextmanager
from mcp.server.fastmcp import FastMCP

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


#Initialize the MCP server
mcp = FastMCP("Tool Server")


@mcp.tool()
def calculate_percentage(amount: float, percentage: float) -> float:
    """Calculate a percentage of an amount.
    
    Args:
        amount: The base amount to calculate percentage of
        percentage: The percentage to calculate (e.g., 15 for 15%)
    
    Returns:
        The calculated percentage amount
    """
    print(f"Calculating {percentage}% of {amount}")
    return (amount * percentage) / 100

@mcp.tool()
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current date and time in YYYY-MM-DD HH:MM:SS format
    """
    print("Getting current time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def get_weather(city: str) -> dict:
    """Retrieve current weather information for a specified city using the OpenWeatherMap API.

    Args:
        city: The name of the city to query.

    Returns:
        Weather data as a dict or an error message string.
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return "Error: OPENWEATHER_API_KEY environment variable not set."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return f"Error: Could not fetch weather data. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@mcp.tool()
def pdf_report_generator(text: str, file_name: str) -> str:
    """Generates a PDF report from the provided text and saves it to the specified file name.

    Args:
        text: The content to be included in the PDF report.
        file_name: The name of the PDF file to be created.

    Returns:
        Success message with file path or error message.
    """
    try:
        if not file_name.endswith('.pdf'):
            file_name += '.pdf'
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=TA_CENTER)
        body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11, spaceAfter=12, alignment=TA_LEFT, leftIndent=20)
        story.append(Paragraph("Research Report", title_style))
        story.append(Spacer(1, 20))
        for para in text.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip().replace('\n', ' '), body_style))
                story.append(Spacer(1, 10))
        doc.build(story)
        return f"PDF report successfully saved as '{file_name}' at {os.path.abspath(file_name)}"
    except Exception as e:
        return f"Error creating PDF: {str(e)}"


@mcp.resource("file://test_doc.txt")
def read_document() -> str:
    with open("documents/test_doc.txt", "r") as f:
        return f.read()

@mcp.resource("financialreport://my_financial_report.pdf")
def read_financial_report() -> str:
    with open("documents/2013_Consolidated_Financial_Report.pdf", "rb") as f:
        return f.read() 


@mcp.prompt(title="helpful_assistant")
def helpful_assistant() -> str:
    prompt = f"""
    You are a helpful Agent who can answer user's questions and perform tasks.
    You can use the following tools to perform tasks:
    - get_weather: Get the current weather for a city using the OpenWeatherMap API
    - get_current_time: Get the current date and time
    - calculate_percentage: Calculate a percentage of an amount
    - pdf_report_generator: Generate a PDF report from text content
    - search_tool: Search the web for information, and any real time information like news, events, weather, etc.
    - wikipedia: Search Wikipedia for information

    """
    return prompt


#lifespac says what should happen in lifecycle of the server
#lifecycle -- start --- running --- shut down
#Session manager -- will hep to manage session for each client
@asynccontextmanager
async def lifespan(app):
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(mcp.session_manager.run())
        yield


app = Starlette(
    debug=True,
    routes=[
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)