import os
from groq import Groq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
# Configure API key from file
os.environ['GROQ_API_KEY'] = os.getenv('groq_key')

# Configure Google GenAI
client = Groq()

# Define system prompt
system_prompt = 'Generate the response very concisely within 2 to 3 points'

st.title('Data Science Tutor application')

# Get user query input
user_prompt = st.text_input('Enter your query:', placeholder='Type your query here . . .')

# When button is clicked, generate and display the response
if st.button('Answer'):
    if user_prompt.strip() != "":
        response = client.chat.completions.create(
            model = 'qwen/qwen3-32b', 
            messages = [
                {'role': 'user', 'content': user_prompt},
                {'role': 'system', 'content': system_prompt}
            ]
        )
        st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter a query before generating an answer.")