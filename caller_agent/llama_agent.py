from vllm import LLM, SamplingParams
from typing import List, Optional
import logging
import time
import os


template = """[INST] <<SYS>>
You are Emma, a sales representative calling on behalf of Neemans Shoes. Your goal is to deliver a sales pitch to the User over the phone.
Use the conversation history as context. Craft a response that effectively addresses the user's needs and preferences.

Conversation History:
{history}

<</SYS>>
User Input (Phone Call): {msg}
<<SYS>>Please ensure your response is short and succinct. Response should be than 1 sentence long.<</SYS>
[/INST]
"""


class LlamaAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.conversation_history = []

    def get_response(self, user_input: str, sampling_params: SamplingParams = SamplingParams(temperature=0.1, top_p=0.90, top_k=20, max_tokens=128)):
        history_text = "\n".join(
            [f"User: {entry['user_input']}\nAssistant: {entry['assistant_response']}" for entry in self.conversation_history])

        prompt = template.format(msg=user_input, history=history_text)

        try:
            start = time.time()
            outputs = self.llm.generate(
                [prompt], sampling_params, use_tqdm=False)
            end = time.time()

            generated_text = outputs[0].outputs[0].text
            conversation_entry = {"user_input": user_input,
                                  "assistant_response": generated_text}
            self.conversation_history.append(conversation_entry)
            logging.info(
                f"Generated response in {end-start:.2f}s")
            return generated_text,
        except Exception as e:
            logging.error(
                f"An error occurred during response generation: {str(e)}")
            return "An error occurred during response generation."
