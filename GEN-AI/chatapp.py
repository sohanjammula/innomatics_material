import os
from google import genai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load API key safely
os.environ['GOOGLE_API_KEY'] = os.getenv("gemini_key")

# Initialize client once in session_state
if "client" not in st.session_state:
    st.session_state.client = genai.Client()

client = st.session_state.client

system_prompt = """You are an expert in Generative AI and recent trends.
Answer queries about Generative AI internships with atmost 2 sentences.
"""

st.title("Gemini Chatbot")
st.write("Type your message below to chat with the model.")

# Initialize chat session only once
if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

print(st.session_state)

# Display past messages
for role, text in st.session_state.messages:
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # Send message safely
    chat = st.session_state.chat_session
    response = chat.send_message(user_input)

    bot_reply = response.text
    st.session_state.messages.append(("bot", bot_reply))

    st.rerun()