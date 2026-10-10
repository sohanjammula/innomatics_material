from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
import os

llm = init_chat_model("openai/gpt-oss-120b", 
                      model_provider="openai",
                      base_url="https://api.groq.com/openai/v1",
                      api_key = os.getenv("GROQ_API_KEY"))
prompt =  ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, who answers user questions"),
    ("user", "{question}")
])


chain = prompt | llm  #chain
while True: #infinite loop
    question = input("You: ") #input -- take question/input through terminal
    if question.lower() == "exit":
        break
    response = chain.invoke({"question": question})
    print("Bot:", response.content)