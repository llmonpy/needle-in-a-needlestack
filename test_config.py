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
from llm_client import GPT3_5, \
    ANTHROPIC_SONNET, MISTRAL_7B,  MISTRAL_8X22B, MISTRAL_SMALL, \
    GPT4, ANTHROPIC_HAIKU

EVALUATOR_MODEL_LIST = [ANTHROPIC_HAIKU, MISTRAL_SMALL, MISTRAL_8X22B, MISTRAL_8X22B,
                        MISTRAL_SMALL]
TEST_DIRECTORY = "tests"


class TestConfig:
    def __init__(self, model_list, evaluator_model_list, test_thread_count,
                 number_of_questions_per_trial, repeat_question_limerick_count, trial_count, location_count):
        self.model_list = model_list
        self.evaluator_model_list = evaluator_model_list
        self.test_thread_count = test_thread_count
        self.number_of_questions_per_trial = number_of_questions_per_trial
        self.repeat_question_limerick_count = repeat_question_limerick_count
        self.trial_count = trial_count
        self.location_count = location_count

    def get_model(self, model_name):
        result = None
        for model in self.model_list:
            if model.model_name == model_name:
                result = model
                break
        return result


DEFAULT_TEST_CONFIG = TestConfig(model_list=[MISTRAL_7B],
                                 test_thread_count=100,
                                 evaluator_model_list=EVALUATOR_MODEL_LIST,
                                 number_of_questions_per_trial=5,
                                 repeat_question_limerick_count=1,
                                 trial_count=5,
                                 location_count=5)


CURRENT_TEST_CONFIG = DEFAULT_TEST_CONFIG
