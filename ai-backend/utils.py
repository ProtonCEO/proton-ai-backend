# utils.py

def build_prompt(prompt, history, user_input):
    conversation = f"{prompt.strip()}\n"
    for entry in history:
        user_msg = entry.get('user', '').strip()
        bot_msg = entry.get('bot', '').strip()
        conversation += f"User: {user_msg}\nBot: {bot_msg}\n"
    conversation += f"User: {user_input.strip()}\nBot:"
    return conversation
