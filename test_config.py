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
import os

from evaluator import DefaultEvaluator
from llm_client import GPT3_5, \
    ANTHROPIC_SONNET, MISTRAL_7B, MISTRAL_8X22B, MISTRAL_SMALL, \
    GPT4, ANTHROPIC_HAIKU, DEEPSEEK, MISTRAL_LARGE, MISTRAL_8X7B, ANTHROPIC_OPUS, GPT4o

EVALUATOR_MODEL_LIST = [ANTHROPIC_HAIKU, MISTRAL_SMALL, MISTRAL_8X22B, MISTRAL_8X22B,
                        MISTRAL_SMALL]
TEST_DIRECTORY = "tests"


class TestConfig:
    def __init__(self, model_list, evaluator_model_list, default_evaluator, test_thread_count,
                 number_of_questions_per_trial, repeat_question_limerick_count, trial_count, location_count):
        self.model_list = model_list
        self.evaluator_model_list = evaluator_model_list
        self.default_evaluator = default_evaluator
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


DEFAULT_TEST_CONFIG = TestConfig(model_list=[GPT3_5],
                                 test_thread_count=100,
                                 evaluator_model_list=EVALUATOR_MODEL_LIST,
                                 default_evaluator=DefaultEvaluator(EVALUATOR_MODEL_LIST),
                                 number_of_questions_per_trial=10,
                                 repeat_question_limerick_count=1,
                                 trial_count=10,
                                 location_count=10)


CURRENT_TEST_CONFIG = DEFAULT_TEST_CONFIG

def get_latest_test_directory():
    directories = [os.path.join(TEST_DIRECTORY, d) for d in os.listdir(TEST_DIRECTORY) if
                   os.path.isdir(os.path.join(TEST_DIRECTORY, d))]
    result = max(directories, key=os.path.getmtime)
    return result