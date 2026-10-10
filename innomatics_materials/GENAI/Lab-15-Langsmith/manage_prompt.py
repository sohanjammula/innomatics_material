from langsmith import Client
from langchain_core.prompts.structured import StructuredPrompt
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

class ResponseSchema(BaseModel):
    positive_sentiment: bool = Field(description="Was the user sentiment positive?")

data_analysis_prompt = """
You are a skilled Data Analyst. 
You have access to a tool called execute_pandas_code that allows you to run only one line pandas code on the provided CSV file to analyze and extract insights.
Use this tool to explore the data, perform calculations, and answer the user's questions accurately and concisely based on the CSV file.

Do not use any other tools or information than the provided CSV file to answer the user's question.

Focus your pandas code analysis strictly on what is required to answer the user's question, rather than exploring the entire dataset.
You should write only one line of pandas code to analyze the data.
Example: df.head() or df.describe() or df.groupby('column').sum()
"""

human_prompt = """
{user_input}

CSV file:
"Churn_Modelling.csv"
"""

prompt = StructuredPrompt.from_messages_and_schema(
    [
        ("system", data_analysis_prompt),
        ("human", human_prompt),
    ],
    schema=ResponseSchema.model_json_schema(),
)

client = Client()
url = client.push_prompt("pandas_agent", object=prompt)
print(url)