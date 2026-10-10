import streamlit as st


from chatbot_redis_memory import (
    get_chain_with_history,
    get_redis_history,
)

st.title("Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = ""

with st.sidebar:
    new_session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        st.rerun()


chain = get_chain_with_history()
config = {"configurable": {"session_id": st.session_state.session_id}}

history_messages = get_redis_history(st.session_state.session_id).messages

for msg in history_messages:
    if msg.type == "system":
        continue
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)

    response = chain.invoke({"input": prompt}, config=config)

    with st.chat_message("assistant"):
        st.write(response.content)
