import os

import anthropic
import openai
from openai import OpenAI


class LlmClient:
    def __init__(self, llm_name, max_input):
        self.llm_name = llm_name
        self.max_input = max_input

    def prompt(self, prompt_text):
        raise Exception("Not implemented")

    def in_context_window(self, location):
        result = location < self.max_input
        return result


class OpenAIModel(LlmClient):
    def __init__(self, llm_name, max_input):
        super().__init__(llm_name, max_input)
        self.client = OpenAI()

    def prompt(self, prompt_text):
        completion = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result


class AnthropicModel(LlmClient):
    def __init__(self, llm_name, max_input):
        super().__init__(llm_name, max_input)
        key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Client()

    def prompt(self, prompt_text):
        message = self.client.messages.create(
            model=self.llm_name,
            max_tokens=4096,
            temperature=0.0,
            system="You are an expert at analyzing text.",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        result = message.content[0].text
        return result

GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 16000)
GPT4 = OpenAIModel('gpt-4-0125-preview', 128000)
ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 200000)
ANTHROPIC_SONNET = AnthropicModel("claude-3-sonnet-20240229", 200000)
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 200000)

LLM_CLIENT_LIST = [GPT3_5, ANTHROPIC_HAIKU]