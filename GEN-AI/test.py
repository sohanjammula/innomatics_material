import os
from openai import OpenAI
from google import genai
from mistralai import Mistral
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_key')
gemini_client = genai.Client()
print(gemini_client)

os.environ['OPENAI_API_KEY'] = os.getenv('openai_key')
openai_client = OpenAI()
print(openai_client)

os.environ['MISTRAL_API_KEY'] = os.getenv('mistral_key')
mistral_client = Mistral()
print(mistral_client)

os.environ['GROQ_API_KEY'] = os.getenv('groq_key')
groq_client = Groq()
print(groq_client)

groq2_client = OpenAI(api_key=os.getenv("groq_key"),
                      base_url="https://api.groq.com/openai/v1")
print(groq2_client)