import os
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from rag import get_answer

load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")

# Webhook verification (Meta requirement)
@app.route("/webhook", methods=["GET"])
def verify():
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if token == VERIFY_TOKEN:
        return challenge
    return "Verification failed", 403


# Handle incoming messages
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_number = message["from"]
        user_text = message["text"]["body"]

        print(f"User: {user_text}")

        # Get AI response from RAG
        response_text = get_answer(user_text)

        send_message(user_number, response_text)

    except Exception as e:
        print("Error:", e)

    return jsonify({"status": "ok"}), 200


def send_message(to, text):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    response = requests.post(url, headers=headers, json=payload)
    print("WhatsApp response:", response.text)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
