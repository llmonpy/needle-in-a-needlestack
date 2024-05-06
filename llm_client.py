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

from rate_llmiter import RateLlmiter, SECOND_TIME_WINDOW, spread_requests

PROMPT_RETRIES = 3
RATE_LIMIT_RETRIES = 100 # requests limits tend to be much higher than token limits, so can end up with a lot of retries
BASE_RETRY_DELAY = 30 # seconds


LIMERICK_PART_API_PREFIX = "LIMERICK_PARK_"


def get_api_key(api_name):
    key = os.environ.get(LIMERICK_PART_API_PREFIX + api_name)
    if key is None:
        key = os.environ.get(api_name)
    return key


def backoff_after_exception(attempt):
    delay_time = (attempt + 1) * BASE_RETRY_DELAY
    time.sleep(delay_time)


class LlmClientRateLimitException(Exception):
    def __init__(self):
        super().__init__("Rate limit exceeded")
        self.status_code = 429

class LlmClient:
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        self.model_name = model_name
        self.max_input = max_input
        self.rate_limiter = rate_limiter
        self.thread_pool = thead_pool

    def prompt(self, prompt_text, system_prompt=None):
        result = None
        self.rate_limiter.get_ticket()
        for attempt in range(RATE_LIMIT_RETRIES):
            try:
                result = self.do_prompt(prompt_text, system_prompt)
                if result is None or len(result) == 0:
                    # some llms return empty result when the rate limit is exceeded, throw exception to retry
                    raise LlmClientRateLimitException()
                else:
                    break
            except Exception as e:
                if getattr(e,"status_code", None) is not None and e.status_code == 429:
                    self.rate_limiter.wait_for_ticket_after_rate_limit_exceeded()
                    continue
                else:
                    raise e
        if result is None:
            raise LlmClientRateLimitException()
        return result

    def do_prompt(self, prompt_text, system_prompt=None):
        raise Exception("Not implemented")

    def get_eval_executor(self):
        return self.thread_pool


class OpenAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result


class AnthropicModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        message = self.client.messages.create(
            model=self.model_name,
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
    def __init__(self, model_name, max_input, rate_limiter, thead_pool):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("MISTRAL_API_KEY")
        self.client = MistralClient(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        prompt_messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user",content=prompt_text)
        ]
        response = self.client.chat(
            model=self.model_name,
            max_tokens=1024,
            temperature=0.0,
            messages=prompt_messages
        )
        result = response.choices[0].message.content
        return result


class OllamaModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        response = ollama.generate(
            model=self.model_name,
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

MISTRAL_RATE_LIMITER = RateLlmiter(*spread_requests(1000)) #used minute spread to seconds because tokens are TPM not TPS

#MIXTRAL tokenizer generates 20% more tokens than openai, so after reduce max_input to 80% of openai
MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 50000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_SMALL = MistralLlmClient("mistral-small", 25000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_7B = MistralLlmClient("open-mistral-7b", 12000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_8X7B = MistralLlmClient("open-mixtral-8x7b", 12000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 15000, RateLlmiter(*spread_requests(2500)), OPENAI_EXECUTOR)
GPT4 = OpenAIModel('gpt-4-turbo-2024-04-09', 15000, RateLlmiter(*spread_requests(2500)), OPENAI_EXECUTOR)
ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 195000, RateLlmiter(*spread_requests(1000)), ANTHROPIC_EXECUTOR)
ANTHROPIC_SONNET = AnthropicModel("claude-3-sonnet-20240229", 195000, RateLlmiter( *spread_requests(1000)), ANTHROPIC_EXECUTOR)
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 15000, RateLlmiter(*spread_requests(1000)), ANTHROPIC_EXECUTOR)

