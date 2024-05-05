# Copyright
import concurrent
import os
import threading
import time
from queue import Queue, Empty

import anthropic
import ollama
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI

PROMPT_RETRIES = 3
BASE_RETRY_DELAY = 30 # seconds

MINUTE_TIME_WINDOW = 60
SECOND_TIME_WINDOW = 1

LIMERICK_PART_API_PREFIX = "LIMERICK_PARK_"


def get_api_key(api_name):
    key = os.environ.get(LIMERICK_PART_API_PREFIX + api_name)
    if key is None:
        key = os.environ.get(api_name)
    return key


def backoff_after_exception(attempt):
    delay_time = (attempt + 1) * BASE_RETRY_DELAY
    time.sleep(delay_time)


class RateLimiterRequestBucket:
    def __init__(self, name, rate_limit, time_window):
        self.name = name
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.queue = Queue()
        self.add_tokens()
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def add_tokens(self):
        while not self.queue.empty():
            self.queue.get_nowait()
        for _ in range(self.rate_limit):
            self.queue.put("token")
        self.timer = threading.Timer(interval=self.time_window, function=self.add_tokens)
        self.timer.start()

    def get_token(self):
        try:
            result = self.queue.get(timeout=(60*10))
        except Empty:
            print("could not get token " + self.name)
            raise Exception("Rate limit exceeded")
        return result


class LlmClient:
    def __init__(self, llm_name, max_input, eval_executor=None):
        self.llm_name = llm_name
        self.max_input = max_input
        self.eval_executor = eval_executor

    def prompt(self, prompt_text, system_prompt=None):
        raise Exception("Not implemented")

    def get_eval_executor(self):
        return self.eval_executor


class OpenAIModel(LlmClient):
    def __init__(self, llm_name, max_input, rate_limiter, eval_executor=None):
        super().__init__(llm_name, max_input, eval_executor)
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
    def __init__(self, llm_name, max_input, rate_limiter, eval_executor=None):
        super().__init__(llm_name, max_input, eval_executor)
        key = get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=key)
        self.rate_limiter = rate_limiter

    def prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        self.rate_limiter.get_token()
        message = self.client.messages.create(
            model=self.llm_name,
            max_tokens=1024,
            temperature=0.0,
            top_k=1,
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
    def __init__(self, llm_name, max_input, rate_limiter, eval_executor):
        super().__init__(llm_name, max_input, eval_executor)
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


class OllamaModel(LlmClient):
    def __init__(self, llm_name, max_input, rate_limiter, eval_executor=None):
        super().__init__(llm_name, max_input, eval_executor)
        self.rate_limiter = rate_limiter

    def prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        self.rate_limiter.get_token()
        response = ollama.generate(
            model=self.llm_name,
            stream=False,
            system_prompt=system_prompt,
            prompt=prompt_text,
            temperature=0.0
        )
        result = response.choices[0].message.content
        return result

OLLAMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
MISTRAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
ANTHROPIC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
OPENAI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)

MISTRAL_EVAL_RATE_LIMITER = RateLimiterRequestBucket("mistral_eval", 10, SECOND_TIME_WINDOW)
MISTRAL_TEST_RATE_LIMITER = RateLimiterRequestBucket("mistral_test", 8, SECOND_TIME_WINDOW)

EVAL_MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 63000, MISTRAL_EVAL_RATE_LIMITER, MISTRAL_EXECUTOR)
EVAL_MISTRAL_SMALL = MistralLlmClient("mistral-small", 25000, MISTRAL_EVAL_RATE_LIMITER, MISTRAL_EXECUTOR)
EVAL_MISTRAL_7B = MistralLlmClient("open-mistral-7b", 12000, MISTRAL_EVAL_RATE_LIMITER, MISTRAL_EXECUTOR)
EVAL_MISTRAL_8X7B = MistralLlmClient("open-mixtral-8x7b", 12000, MISTRAL_EVAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 51000, MISTRAL_EVAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_7B = MistralLlmClient("open-mistral-7b", 12000, MISTRAL_TEST_RATE_LIMITER, MISTRAL_EXECUTOR)
#MIXTRAL tokenizer seems to generate a lot more tokens than openai
MIXTRAL_8X7B = MistralLlmClient("open-mixtral-8x7b", 25000, MISTRAL_TEST_RATE_LIMITER, MISTRAL_EXECUTOR)
EVAL_GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 15000, RateLimiterRequestBucket("open_ai_35_eval", 150, MINUTE_TIME_WINDOW), OPENAI_EXECUTOR)
EVAL_GPT4 = OpenAIModel('gpt-4-turbo-2024-04-09', 127000, RateLimiterRequestBucket("open_ai_4_eval", 150, MINUTE_TIME_WINDOW), OPENAI_EXECUTOR)
GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 15000, RateLimiterRequestBucket("open_ai_35_test", 40, MINUTE_TIME_WINDOW), OPENAI_EXECUTOR)
GPT4 = OpenAIModel('gpt-4-turbo-2024-04-09', 15000, RateLimiterRequestBucket("open_ai_4_test", 40, MINUTE_TIME_WINDOW), OPENAI_EXECUTOR)
EVAL_ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 199000, RateLimiterRequestBucket("ant_o_eval", 100, MINUTE_TIME_WINDOW), ANTHROPIC_EXECUTOR)
EVAL_ANTHROPIC_SONNET = AnthropicModel("claude-3-sonnet-20240229", 199000, RateLimiterRequestBucket("ant_s_eval", 100, MINUTE_TIME_WINDOW), ANTHROPIC_EXECUTOR)
EVAL_ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 199000, RateLimiterRequestBucket("ant_h_eval", 200, MINUTE_TIME_WINDOW), ANTHROPIC_EXECUTOR)
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 16000, RateLimiterRequestBucket("ant_h_eval", 20, MINUTE_TIME_WINDOW), ANTHROPIC_EXECUTOR)

