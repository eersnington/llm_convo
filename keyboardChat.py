from gevent import monkey

monkey.patch_all()

from vllm import LLM
from caller_agent.agents import LlamaChatAgent, TerminalInPrintOut
from caller_agent.conversation import run_conversation


def main():
    agent_a = LlamaChatAgent(llm=LLM(model="TheBloke/Llama-2-7B-chat-AWQ", quantization="awq"), init_phrase="Hello?")
    agent_b = TerminalInPrintOut()
    run_conversation(agent_a, agent_b)


if __name__ == "__main__":
    main()
