import time
import streamlit as st
from main import setup_vector_store
from chatbot import create_rag_chain

st.set_page_config(page_title="Equipment Troubleshooting Assistant")


@st.cache_resource
def initialize_chatbot():
    setup_vector_store()
    return create_rag_chain()


st.title("Equipment Troubleshooting Assistant")
st.caption("Ask about installation, maintenance, or troubleshooting for the conveyor manuals in the knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

try:
    chatbot = initialize_chatbot()
except Exception as e:
    st.error(f"Failed to initialize the chatbot: {e}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about the equipment manuals...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
