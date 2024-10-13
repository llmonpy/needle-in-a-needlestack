#  Copyright © 2024 Thomas Edward Burns
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#  Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import concurrent
import os
import threading
import time
from queue import Queue, Empty

import anthropic
import ollama
from ai21 import AI21Client
from ai21.models.chat import SystemMessage, UserMessage
from fireworks.client import Fireworks
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openai import OpenAI
import google.generativeai as genai
from ratellmiter.rate_llmiter import BucketRateLimiter, llmiter, RateLimitedService

PROMPT_RETRIES = 3
RATE_LIMIT_RETRIES = 100  # requests limits tend to be much higher than token limits, so can end up with a lot of retries
BASE_RETRY_DELAY = 30  # seconds

NIAN_API_PREFIX = "NIAN_"

gVERTEX_AI_INITED = False

'''
def init_vertex_ai():
    global gVERTEX_AI_INITED
    if not gVERTEX_AI_INITED:
        PROJECT_ID = os.environ.get("GCLOUD_PROJECT")
        REGION = os.environ.get("GCLOUD_REGION")
        vertexai.init(project=PROJECT_ID, location=REGION)
    gVERTEX_AI_INITED = True
'''


def get_api_key(api_name, exit_on_error=True):
    key = os.environ.get(NIAN_API_PREFIX + api_name)
    if key is None:
        key = os.environ.get(api_name)
    if key is None and exit_on_error:
        print("API key not found for " + api_name)
        exit(1)
    return key


def backoff_after_exception(attempt):
    delay_time = (attempt + 1) * BASE_RETRY_DELAY
    time.sleep(delay_time)


class LlmClientRateLimitException(Exception):
    def __init__(self):
        super().__init__("Rate limit exceeded")
        self.status_code = 429


class LlmClient(RateLimitedService):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        self.model_name = model_name
        self.max_input = max_input
        self.rate_limiter = rate_limiter
        if self.rate_limiter is not None:
            rate_limiter.set_rate_limited_service(self)
        self.thread_pool = thead_pool

    def get_ratellmiter(self, model_name=None):
        return self.rate_limiter

    def ratellmiter_is_llm_blocked(self):
        result = True
        print("Testing if blocked")
        try:
            self.do_prompt("What is the capital of France?","You are a helpful assistant")
            result = False
            print(self.model_name + " is not blocked")
        except Exception as e:
            print("Blocked test exception: " + str(e))
            result = True
        return result

    def get_service_name(self):
        return self.model_name


    @llmiter(debug=False)
    def prompt(self, prompt_text, system_prompt=None):
        result = None
        result = self.do_prompt(prompt_text, system_prompt)
        return result

    '''
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
                if getattr(e, "status_code", None) is not None and e.status_code == 429:
                    self.rate_limiter.wait_for_ticket_after_rate_limit_exceeded()
                    continue
                else:
                    raise e
        if result is None:
            raise LlmClientRateLimitException()
        return result
    '''
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
            temperature=0.0,
            timeout=90
        )
        result = completion.choices[0].message.content
        return result


class DeepseekModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com/")

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
            ChatMessage(role="user", content=prompt_text)
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


# https://ai.google.dev/gemini-api/docs/get-started/tutorial?authuser=2&lang=python
class GeminiModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.client = genai.GenerativeModel(self.model_name)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        full_prompt = system_prompt + "\n\n" + prompt_text
        model_response = self.client.generate_content(full_prompt,
                                                      safety_settings={
                                                          HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                          HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                          HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                          HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                                      },
                                                      generation_config=genai.GenerationConfig(temperature=0.0))
        result = model_response.text
        return result


class FireworksAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None, system_role_supported=True):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("FIREWORKS_API_KEY")
        self.client = Fireworks(api_key=key)
        self.system_role_supported = system_role_supported

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        if self.system_role_supported:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
            )
        else:
            full_prompt = str(system_prompt) + "\n\n" + prompt_text
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
            )
        result = completion.choices[0].message.content
        return result


class AI21Model(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("AI21_API_KEY")
        self.client = AI21Client(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.0,
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=prompt_text)
            ],
        )
        result = completion.choices[0].message.content
        return result


OLLAMA_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
MISTRAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
ANTHROPIC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
OPENAI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
DEEPSEEK_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
GEMINI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
FIREWORKS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
AI21_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)
HYPERBOLIC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=250)

MISTRAL_RATE_LIMITER = BucketRateLimiter(300, "MISTRAL")
FIREWORKS_RATE_LIMITER = BucketRateLimiter(480, "FIREWORKS")
AI21_RATE_LIMITER = BucketRateLimiter(60, "AI21")

# MIXTRAL tokenizer generates  20% more tokens than openai, so after reduce max_input to 80% of openai
MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 8000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_SMALL = MistralLlmClient("mistral-small-2409", 20000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_7B = MistralLlmClient("open-mistral-7b", 20000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_NEMO_12B = MistralLlmClient("open-mistral-nemo-2407", 32000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_8X7B = MistralLlmClient("open-mixtral-8x7b", 24000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_LARGE = MistralLlmClient("mistral-large-latest", 24000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
MISTRAL_LARGE2 = MistralLlmClient("mistral-large-2407", 32000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 12000, BucketRateLimiter(5000), OPENAI_EXECUTOR)
GPT4 = OpenAIModel('gpt-4-turbo-2024-04-09', 16000, BucketRateLimiter(5000), OPENAI_EXECUTOR)
GPT4o = OpenAIModel('gpt-4o', 12000, BucketRateLimiter(10000), OPENAI_EXECUTOR)
GPT4omini = OpenAIModel('gpt-4o-mini', 12000, BucketRateLimiter(10000), OPENAI_EXECUTOR)
ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 195000, BucketRateLimiter(3),
                                ANTHROPIC_EXECUTOR)
ANTHROPIC_SONNET = AnthropicModel("claude-3-5-sonnet-20240620", 32000, BucketRateLimiter(480),
                                  ANTHROPIC_EXECUTOR)
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 12000, BucketRateLimiter(480),
                                 ANTHROPIC_EXECUTOR)
GEMINI_FLASH = GeminiModel("gemini-1.5-flash-002", 12000, BucketRateLimiter(1200), GEMINI_EXECUTOR)
GEMINI_FLASH_8B = GeminiModel("gemini-1.5-flash-8b", 12000, BucketRateLimiter(1200), GEMINI_EXECUTOR)
GEMINI_PRO = GeminiModel("gemini-1.5-pro-002", 120000, BucketRateLimiter(10), GEMINI_EXECUTOR)
FIREWORKS_LLAMA3_2_1B = FireworksAIModel("accounts/fireworks/models/llama-v3p2-1b-instruct", 4000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_EXECUTOR)
FIREWORKS_LLAMA3_2_3B = FireworksAIModel("accounts/fireworks/models/llama-v3p2-3b-instruct", 8000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_EXECUTOR)
FIREWORKS_LLAMA3_1_8B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-8b-instruct", 8000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_EXECUTOR)
FIREWORKS_LLAMA3_1_405B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-405b-instruct", 4000,
                                           FIREWORKS_RATE_LIMITER, FIREWORKS_EXECUTOR)
FIREWORKS_LLAMA3_1_70B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-70b-instruct", 4000,
                                          FIREWORKS_RATE_LIMITER, FIREWORKS_EXECUTOR)
AI21_JAMBA_1_5_MINI = AI21Model("jamba-1.5-mini", 12000,
                                AI21_RATE_LIMITER, AI21_EXECUTOR)


