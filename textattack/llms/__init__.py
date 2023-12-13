"""
Large Language Models
======================

TextAttack can generate responses to prompts using LLMs, which take in a list of strings and outputs a list of responses.

We've provided an implementation around two common LLM patterns:

1. `HuggingFaceLLMWrapper` for LLMs in HuggingFace
2. `ChatGptWrapper` for OpenAI's ChatGPT model


"""

from .chat_gpt_wrapper import ChatGptWrapper
from .huggingface_llm_wrapper import HuggingFaceLLMWrapper
