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
import copy
import json
import os
import random

from limerick import Limerick, FULL_QUESTION_FILE, read_and_init_limericks, LIMERICK_DATASET_FILE
from test_config import TEST_DIRECTORY

PROMPT_FILE_NAME = "test_prompt"

INTRO_TO_PROMPT = "This is a test to see how well you are paying attention. This text is a series of limericks. " \
    "At the end of the list of limericks, there will be a question. The question will be about one of the limericks. " \
    "Please answer the question as concisely as possible. "

class LimerickPrompt:
    def __init__(self, target_size, question_list, text, token_count=0, limerick_list=None):
        self.target_size = target_size
        self.question_list = question_list
        self.limerick_list = limerick_list
        self.text = text
        self.token_count = token_count

    def add_limerick(self, limerick):
        if self.token_count + limerick.token_count <= self.target_size:
            if self.limerick_list is None:
                self.limerick_list = []
            self.limerick_list.append(limerick)
            self.token_count += limerick.token_count

    def build_text_from_limerick_list(self, question, location, max_size, repeat_question_count=1):
        result = None
        limerick_used_count = 0
        last_token_count = current_token_count = 0
        result = self.text + "\n\n" # intro of prompt was added in the constructor
        added_question = False
        for limerick in self.limerick_list:
            limerick_used_count += 1
            current_token_count += limerick.token_count
            if current_token_count > max_size:
                break
            if last_token_count < location <= current_token_count:
                for i in range(repeat_question_count):
                    result += "\n\n" + question.text
                    current_token_count += question.token_count
                added_question = True
            result += "\n\n" + limerick.text
            last_token_count = current_token_count
        if not added_question:
            raise Exception("Question was not added to prompt")
        return result, limerick_used_count

    def write_to_file(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.question_list is not None:
            index = 0
            for question in self.question_list:
                result["question_list"][index] = question.to_dict()
                index += 1
        if self.limerick_list is not None:
            index = 0
            for limerick in self.limerick_list:
                result["limerick_list"][index] = limerick.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        question_list = dictionary.get("question_list", None)
        if question_list is not None:
            dictionary.pop("question_list", None)
            question_list = [Limerick.from_dict(question_dict) for question_dict in question_list]
            dictionary["question_list"] = question_list
        limerick_list = dictionary.get("limerick_list", None)
        if limerick_list is not None:
            dictionary.pop("limerick_list", None)
            limerick_list = [Limerick.from_dict(limerick_dict) for limerick_dict in limerick_list]
            dictionary["limerick_list"] = limerick_list
        result = LimerickPrompt(**dictionary)
        return result

    @staticmethod
    def for_target_size(target_size, question_list):
        result = LimerickPrompt(target_size, copy.copy(question_list), INTRO_TO_PROMPT)
        return result


class LimerickListBuilder:
    def __init__(self, question_dict):
        self.limerick_list = []
        self.limerick_dict = {}
        self.question_dict = question_dict
        self.current_token_count = 0
        self.prior_token_count = 0

    def test_and_add_limerick(self, limerick):
        if self.limerick_dict.get(limerick.id, None) is None and self.question_dict.get(limerick.id, None) is None:
            self.add_limerick(limerick)

    def add_limerick(self, limerick):
        self.prior_token_count = self.current_token_count
        self.limerick_dict[limerick.id] = limerick
        self.limerick_list.append(limerick)
        self.current_token_count += limerick.token_count


def select_questions_for_prompt(file_path, number_of_questions):
    with open(file_path, "r") as file:
        question_dict_list = json.load(file)
    question_list = [Limerick.from_dict(question_dict) for question_dict in question_dict_list]
    selected_question_dict = {}
    while len(selected_question_dict) < number_of_questions:
        index = random.randint(0, len(question_list) - 1)
        question = question_list[index]
        if selected_question_dict.get(question.id, None) is None:
            question = copy.copy(question)
            selected_question_dict[question.id] = question
    result = list(selected_question_dict.values())
    return result, selected_question_dict


def select_limericks_for_prompt(limerick_list, question_dict, max_token_count):
    builder = LimerickListBuilder(question_dict)
    while builder.current_token_count < max_token_count:
        index = random.randint(0, len(limerick_list) - 1)
        limerick = limerick_list[index]
        builder.test_and_add_limerick(limerick)
    result = builder.limerick_list
    return result


def prompt_file_name(base_name, number_of_questions, size):
    result = base_name + "_" + str(number_of_questions) + "_" + str(size) + ".json"
    return result


def generate_prompt(max_size, test_config):
    limerick_list = read_and_init_limericks(LIMERICK_DATASET_FILE)
    selected_question_list, selected_question_dict = select_questions_for_prompt(FULL_QUESTION_FILE,
                                                                                 test_config.number_of_questions_per_trial)
    selected_limerick_list = select_limericks_for_prompt(limerick_list, selected_question_dict,
                                                         max_size)
    result = LimerickPrompt.for_target_size(max_size, selected_question_list)
    index = 0
    print("generating prompt")
    for limerick in selected_limerick_list:
        index += 1
        if index % 10 == 0:
            print(".")
        result.add_limerick(limerick)
    prompt_file_path = os.path.join(TEST_DIRECTORY,
                                    prompt_file_name(PROMPT_FILE_NAME, test_config.number_of_questions_per_trial, max_size))
    result.write_to_file(prompt_file_path)
    return result


def read_prompt(file_name):
    with open(file_name, "r") as file:
        prompt_dict = json.load(file)
        result = LimerickPrompt.from_dict(prompt_dict)
    return result


def get_prompt(max_size, test_config):
    file_name = prompt_file_name(PROMPT_FILE_NAME, test_config.number_of_questions_per_trial, max_size)
    prompt_file_path = os.path.join(TEST_DIRECTORY, file_name)
    result = None
    if os.path.exists(prompt_file_path):
        result = read_prompt(prompt_file_path)
    else:
        generate_prompt(max_size, test_config)
        result = read_prompt(prompt_file_path)
    return result

