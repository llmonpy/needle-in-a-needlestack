
from llm_client import GPT3_5, \
    ANTHROPIC_SONNET, MISTRAL_7B,  MISTRAL_8X22B, MISTRAL_SMALL, \
    GPT4, ANTHROPIC_HAIKU

EVALUATOR_MODEL_LIST = [ANTHROPIC_HAIKU, MISTRAL_SMALL, GPT3_5, MISTRAL_8X22B,
                        MISTRAL_SMALL]


class TestConfig:
    def __init__(self, prompt_file_name, model_list, test_thread_count, evaluator_model_list,
                 number_of_questions_per_trial, repeat_question_limerick_count, trials, location_count,
                 result_directory):
        self.prompt_file_name = prompt_file_name
        self.model_list = model_list
        self.test_thread_count = test_thread_count
        self.evaluator_model_list = evaluator_model_list
        self.number_of_questions_per_trial = number_of_questions_per_trial
        self.repeat_question_limerick_count = repeat_question_limerick_count
        self.trials = trials
        self.location_count = location_count
        self.result_directory = result_directory

    def get_model(self, model_name):
        result = None
        for model in self.model_list:
            if model.model_name == model_name:
                result = model
                break
        return result


DEFAULT_TEST_CONFIG = TestConfig("test_prompt", [MISTRAL_7B], 100, EVALUATOR_MODEL_LIST,
                                 5, 1, 5,
                                 5, "tests")


CURRENT_TEST_CONFIG = DEFAULT_TEST_CONFIG
