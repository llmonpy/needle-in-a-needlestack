# Copyright
import os
import threading
from queue import Queue

import anthropic
import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI

MINUTE_TIME_WINDOW = 60
SECOND_TIME_WINDOW = 1

LIMERICK_PART_API_PREFIX = "LIMERICK_PARK_"


def get_api_key(api_name):
    key = os.environ.get(LIMERICK_PART_API_PREFIX + api_name)
    if key is None:
        key = os.environ.get(api_name)
    return key


class RateLimiterTokenBucket:
    def __init__(self, rate_limit, time_window):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.queue = Queue()
        self.queue_lock = threading.Lock()
        self.add_tokens()
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def add_tokens(self):
        with self.queue_lock:
            while not self.queue.empty():
                self.queue.get_nowait()
            for _ in range(self.rate_limit):
                self.queue.put("token")
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def get_token(self):
        with self.queue_lock:
            result = self.queue.get()
        return result


class LlmClient:
    def __init__(self, llm_name, max_input):
        self.llm_name = llm_name
        self.max_input = max_input

    def prompt(self, prompt_text, system_prompt=None):
        raise Exception("Not implemented")

    def in_context_window(self, location):
        result = location < self.max_input
        return result


class OpenAIModel(LlmClient):
    def __init__(self, llm_name, max_input, rate_limiter):
        super().__init__(llm_name, max_input)
        key = get_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.rate_limiter = rate_limiter

    def prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        self.rate_limiter.get_token()
        completion = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result


class AnthropicModel(LlmClient):
    def __init__(self, llm_name, max_input, rate_limiter):
        super().__init__(llm_name, max_input)
        key = get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=key)
        self.rate_limiter = rate_limiter

    def prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        self.rate_limiter.get_token()
        message = self.client.messages.create(
            model=self.llm_name,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        )
        result = message.content[0].text
        return result

class MistralLlmClient(LlmClient):
    def __init__(self, llm_name, max_input, rate_limiter):
        super().__init__(llm_name, max_input)
        key = get_api_key("MISTRAL_API_KEY")
        self.client = MistralClient(api_key=key)
        self.rate_limiter = rate_limiter

    def prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        self.rate_limiter.get_token()
        prompt_messages = [
            ChatMessage(role="system", content = system_prompt),
            ChatMessage(role="user",content=prompt_text)
        ]
        response = self.client.chat(
            model=self.llm_name,
            max_tokens=1024,
            temperature=0.0,
            messages=prompt_messages
        )
        result = response.choices[0].message.content
        return result

MISTRAL_EVAL_RATE_LIMITER = RateLimiterTokenBucket(3, SECOND_TIME_WINDOW)
MISTRAL_TEST_RATE_LIMITER = RateLimiterTokenBucket(1, SECOND_TIME_WINDOW * 2)

EVAL_MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 32000, MISTRAL_EVAL_RATE_LIMITER)
MISTRAL_7B = MistralLlmClient("open-mistral-7b", 32000, MISTRAL_TEST_RATE_LIMITER)
EVAL_GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 16000, RateLimiterTokenBucket(80, MINUTE_TIME_WINDOW))
EVAL_GPT4 = OpenAIModel('gpt-4-turbo', 128000, RateLimiterTokenBucket(600, MINUTE_TIME_WINDOW))
GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 16000, RateLimiterTokenBucket(15, MINUTE_TIME_WINDOW))
GPT4 = OpenAIModel('gpt-4-turbo', 128000, RateLimiterTokenBucket(4, MINUTE_TIME_WINDOW))
EVAL_ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 200000, RateLimiterTokenBucket(100, MINUTE_TIME_WINDOW))
EVAL_ANTHROPIC_SONNET = AnthropicModel("claude-3-sonnet-20240229", 200000, RateLimiterTokenBucket(400, MINUTE_TIME_WINDOW))
EVAL_ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 200000, RateLimiterTokenBucket(500, MINUTE_TIME_WINDOW))

