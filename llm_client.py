import anthropic
import openai


class LlmClient:
    def __init__(self, llm_name):
        self.llm_name = llm_name

    def prompt(self, prompt_text):
        raise Exception("Not implemented")


class OpenAIModel(LlmClient):
    def __init__(self, llm_name):
        super().__init__(llm_name)

    def prompt(self, prompt_text):
        completion = openai.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result


class AnthropicModel(LlmClient):
    def __init__(self, llm_name):
        super().__init__(llm_name)
        self.client = anthropic.Client()

    def prompt(self, prompt_text):
        message = self.client.messages.create(
            model=self.name,
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

GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125')
GPT4 = OpenAIModel('gpt-4-0125-preview')
ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229")
ANTHROPIC_SONNET = AnthropicModel("claude-3-sonnet-20240229")
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307")

LLM_CLIENTS = [GPT3_5]