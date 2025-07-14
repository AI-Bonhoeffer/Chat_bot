import streamlit as st
from chat_logic import process_user_input
from dotenv import load_dotenv
import re

load_dotenv()
st.set_page_config(page_title="Bonhoeffer Chatbot", page_icon="🤖")

# 🧠 Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_verified" not in st.session_state:
    st.session_state.is_verified = False

# 🖼️ UI
st.title("Bonhoeffer Bot")
user_input = st.chat_input("Type your message...")

# 🧠 Message Processing
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        replies, is_verified = process_user_input(user_input, st.session_state.is_verified)
        st.session_state.is_verified = is_verified  # 🔐 Update verification

        for reply in replies:
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

# 💬 Chat Rendering
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").markdown(f"```\n{msg['content']}\n```")  # Avoid markdown issues
