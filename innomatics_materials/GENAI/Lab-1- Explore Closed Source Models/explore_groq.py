from openai import OpenAI 

from dotenv import load_dotenv

load_dotenv()
import os


client = OpenAI(api_key = os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

prompt = "Hi, How are you?"

#Openai Prompt Structure

#[{"role":"user","content":"Hi, How are you?"}]

response = client.responses.create(
    input= prompt,
    model = "meta-llama/llama-4-scout-17b-16e-instruct",
)
print(response.output_text)