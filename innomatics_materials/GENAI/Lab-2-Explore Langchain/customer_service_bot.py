from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

#RunnablePassthrough  -- LCEL

from dotenv import load_dotenv

load_dotenv()
import os

# Initialize the LLM
llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))


# Step 1: Sentiment Analysis Chain
sentiment_prompt = PromptTemplate.from_template(
    "Analyze sentiment of this complaint/issue: {issue}\n"
    "Respond with only: positive, neutral, or negative"
)

sentiment_chain = sentiment_prompt | llm | StrOutputParser()


# Step 2: Category Classification Chain  
category_prompt = PromptTemplate.from_template(
    "Categorize this issue: {issue}\n"
    "Categories: billing, technical, product, shipping\n"
    "Respond with only the category."
)
category_chain = category_prompt | llm | StrOutputParser()



# Step 3: Email Generation Chain
response_prompt = PromptTemplate.from_template(
    """Write a professional customer service response.
    
Issue: {issue}
Sentiment: {sentiment}
Category: {category}

Be helpful and empathetic."""
)
response_chain = response_prompt | llm | StrOutputParser()



final_chain = (
    RunnablePassthrough.assign(sentiment=sentiment_chain) 
    | RunnablePassthrough.assign(category=category_chain) 
    | RunnablePassthrough.assign(response=response_chain)
)

result = final_chain.invoke({"issue": "I love the product, but the shipping was slow."})
print(result)

print("Sentiment:", result["sentiment"])
print("Category:", result["category"])
print("Response:", result["response"])