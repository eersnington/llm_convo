from vllm import LLM, SamplingParams
from typing import Optional
import logging
import time
import functools


template = """[INST] <<SYS>>
You are Ruby, a customer service agent in Neemans Shoes. User will call you over the phone and your task is to handle queries professionally.
Use the conversation history as context. Craft a response that effectively addresses the user's needs and preferences.

Conversation History:
{history}

<</SYS>>
User Input (Phone Call): {msg}
<<SYS>>Please ensure your response is short and succinct. Your response should ideally just be 1 sentence.<</SYS>
[/INST]
"""


@functools.cache
def get_llm_model():
    logging.info(f"Loading Llama LLM")
    llm = LLM(model="TheBloke/Llama-2-7B-chat-AWQ", quantization="awq")
    return llm


class LlamaAgent:
    def __init__(self, init_phrase: Optional[str] = None):
        self.llm = get_llm_model()
        self.conversation_history = []
        if init_phrase is not None:
            self.conversation_history.append(
                {"user_input": "**NO MESSAGE FROM USER**", "assistant_response": init_phrase})

    def get_response(self, user_input: str, sampling_params: SamplingParams = SamplingParams(temperature=0.1, top_p=0.90, top_k=20, max_tokens=128)) -> str:
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
            return generated_text
        except Exception as e:
            logging.error(
                f"An error occurred during response generation: {str(e)}")
            return "An error occurred during response generation."
