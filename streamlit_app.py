import streamlit as st
import time
from chat_logic import process_user_input
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Bonhoeffer Chatbot", page_icon="🤖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Bonhoeffer Bot")

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        replies, _ = process_user_input(user_input, user_id="streamlit-user")
        for reply in replies:
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
