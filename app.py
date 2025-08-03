# app.py
from flask import Flask, request, Response
from flask_cors import CORS
import time
import os

from model import TinyGPT
from utils import build_prompt  # Assume this builds a full prompt from bot + history + user

app = Flask(__name__)
CORS(app)

model = TinyGPT()

@app.route('/generate', methods=['POST'])
def generate_reply():
    data = request.get_json()
    user_input = data.get('message', '')
    history = data.get('history', [])
    bot_prompt = data.get('prompt', 'You are a helpful assistant.')

    # Build prompt with personality + previous messages
    prompt = build_prompt(bot_prompt, history, user_input)
    reply = model.generate_reply(prompt)

    def generate():
        for word in reply.split():
            yield f"data: {word} \n\n"
            time.sleep(0.03)  # Simulate typing
        yield "data: [END] \n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset_chat():
    model.reset_history()
    return {'status': 'reset'}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
