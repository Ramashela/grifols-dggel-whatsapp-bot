import os
from flask import Flask, request
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

# Import your AI function
from rag import get_answer

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Grifols DG Gel WhatsApp Bot is running 🚀"


@app.route("/webhook", methods=["POST"])
def whatsapp_reply():
    try:
        # Incoming message from WhatsApp
        incoming_msg = request.form.get("Body", "").strip()
        sender = request.form.get("From", "")

        print(f"[INCOMING] From: {sender} | Message: {incoming_msg}")

        # Default reply if empty message
        if not incoming_msg:
            reply = "Please send a valid message."
        else:
            # Get AI response from your RAG system
            reply = get_answer(incoming_msg)

        print(f"[REPLY] {reply}")

    except Exception as e:
        print("[ERROR]", e)
        reply = "⚠️ Sorry, something went wrong. Please try again."

    # Twilio response
    resp = MessagingResponse()
    resp.message(reply)

    return str(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
