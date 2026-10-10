from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

import os
#Init the LLM

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))

#Draft the prompt
# Customer serivve Bot
prompt = PromptTemplate.from_template(
    """
    You are a customer service agent. Take this customer complaint,
    analyze the sentiment, categorize the issue, suggest a solution,
    and write a professional response. Also check our knowledge base
    and escalate if needed...

    Complaint:
    {issue}
    """
)

parser = StrOutputParser()  #String output parser
#Create a chain to process the prompt and get the response from the LLM
# |--Pipeline operator to pass the prompt to the LLM and get the response
chain = prompt | llm | parser

response = chain.invoke({"issue": "I received a damaged product and I want a refund."})

print(response)

