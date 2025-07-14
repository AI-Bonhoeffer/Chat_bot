from flask import Flask, request
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
import os
import time
import re
from db import load_vector_store

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret")

verified_users = {}
vector_store = load_vector_store()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    retriever=vector_store.as_retriever()
)

def process_user_input(user_input, user_id):
    responses = []
    current_time = time.time()
    is_verified = user_id in verified_users and current_time < verified_users[user_id]

    if "7320811109" in user_input and "123456" in user_input:
        verified_users[user_id] = current_time + 86400
        responses.append("✅ You are verified. Valid for 24 hours.")
    elif "7320811109" in user_input or "123456" in user_input:
        responses.append("❌ Wrong ID or password.")
    elif len(user_input.strip()) == 4 and user_input.strip().isalnum():
        if not is_verified:
            responses.append("🔒 Please enter your ID and password to access price info.")
        else:
            query = f"What is the price of model ending with {user_input}?"
            responses.append(qa_chain.run(query))
    elif any(word in user_input.lower() for word in ["production time", "lead time"]):
        responses.append("🏭 Production time is 90 days.")
    elif any(word in user_input.lower() for word in ["price", "cost", "rate"]):
        if not is_verified:
            responses.append("🔒 Please enter your ID and password to access price info.")
        else:
            match = re.search(r"\b([A-Za-z0-9]{4})\b", user_input)
            code = match.group(1) if match else user_input
            responses.append(qa_chain.run(f"What is the price of model ending with {code}?"))
    else:
        responses.append(qa_chain.run(user_input))

    return responses, is_verified

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').strip()
    user_id = request.values.get('From', 'unknown')

    resp = MessagingResponse()
    if not incoming_msg:
        resp.message("⚠️ Sorry, I didn't get your message.")
        return str(resp)

    replies, _ = process_user_input(incoming_msg, user_id)
    for reply in replies:
        resp.message(reply)

    return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
