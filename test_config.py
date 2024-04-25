from llm_client import GPT3_5, EVAL_ANTHROPIC_HAIKU, EVAL_GPT3_5, \
    EVAL_ANTHROPIC_SONNET, EVAL_MISTRAL_8X22B

EVALUATOR_MODEL_LIST = [EVAL_ANTHROPIC_HAIKU, EVAL_GPT3_5, EVAL_GPT3_5, EVAL_MISTRAL_8X22B, EVAL_ANTHROPIC_SONNET]
TEST_MODEL_LIST = [GPT3_5]
class TestConfig:
    def __init__(self, prompt_file_name, model_list, evaluator_model_list, cycles, location_count, result_directory):
        self.prompt_file_name = prompt_file_name
        self.model_list = model_list
        self.evalutor_model_list = evaluator_model_list
        self.cycles = cycles
        self.location_count = location_count
        self.result_directory = result_directory


DEFAULT_TEST_CONFIG = TestConfig("test_prompt", TEST_MODEL_LIST, EVALUATOR_MODEL_LIST,  1,
                                 5, "tests")