# model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TinyGPT:
    def __init__(self):
        print("⚡ Loading distilgpt2 for lightweight performance...")
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.chat_history_ids = None

    def generate_reply(self, prompt):
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt')

        bot_input_ids = (
            torch.cat([self.chat_history_ids, input_ids], dim=-1)
            if self.chat_history_ids is not None
            else input_ids
        )

        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=250,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response

    def reset_history(self):
        self.chat_history_ids = None
