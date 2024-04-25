import concurrent
import copy
import json
import os
import random
from datetime import datetime

from evaluator import DefaultEvaluator
from limerick import Limerick, read_and_init_limericks, FULL_QUESTION_FILE
from llm_client import GPT3_5
from test_config import DEFAULT_TEST_CONFIG, TEST_MODEL_LIST, EVALUATOR_MODEL_LIST

NUMBER_OF_QUESTIONS_PER_PROMPT = 5

INTRO_TO_PROMPT = "This is a test to see how well you are paying attention. This text is a series of limericks. " \
    "At the end of the list of limericks, there will be a question. The question will be about one of the limericks. " \
    "Please answer the question as concisely as possible. "

ROUGH_QUESTION_LOCATIONS = [100, 1200, 5700]

EVALUATOR_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)
TEST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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

    def build_text_from_limerick_list(self, question, location, repeat_count_for_questions=1):
        result = None
        if location < self.token_count:
            last_token_count = current_token_count = 0
            result = self.text + "\n\n" # intro of prompt was added in the constructor
            for limerick in self.limerick_list:
                current_token_count += limerick.token_count
                if last_token_count < location <= current_token_count:
                    for i in range(repeat_count_for_questions):
                        result = result + "\n\n" + question.text
                result += "\n\n" + limerick.text
                last_token_count = current_token_count
        return result

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


def generate_tests(limerick_list, prompt_size_list):
    selected_question_list, selected_question_dict = select_questions_for_prompt(FULL_QUESTION_FILE,
                                                                                 NUMBER_OF_QUESTIONS_PER_PROMPT)
    max_token_count = prompt_size_list[-1]
    selected_limerick_list = select_limericks_for_prompt(limerick_list, selected_question_dict,
                                                         max_token_count)
    prompt_list = [LimerickPrompt.for_target_size(prompt_size, selected_question_list) for prompt_size in prompt_size_list]
    index = 0
    for limerick in selected_limerick_list:
        index += 1
        if index % 10 == 0:
            print(".")
        for prompt in prompt_list:
            prompt.add_limerick(limerick)
    for prompt in prompt_list:
        prompt.write_to_file("test_" + str(prompt.target_size) + ".json")
    return prompt_list


def test_file_name(size):
    result = "test_" + str(size) + ".json"
    return result


def print_result(prompt, client, question, location, result, score):
    print("---------------------------------")
    print("Client:", client.llm_name)
    print("Prompt Size:", prompt.token_count)
    print("Location:", location)
    print("Limerick:", question.text)
    print("Question:", question.question)
    print("Good Answer:", question.answer)
    print("Generated Answer:", result)
    print("Score:", score)


def write_prompt_text_to_file(prompt_text, prompt_file, client_name, location, question_id):
    file_name = prompt_file + "_" + client_name + "_" + location + "_" + question_id + ".txt"
    file_path = os.path.join("full_prompts", file_name)
    with open(file_path, "w") as file:
        file.write(prompt_text)


def run_tests(model, prompt_file_list, question_location_list, evaluator_executor,
              evaluator_model_list, evaluator):
    for prompt_file in prompt_file_list:
        with open(prompt_file, "r") as file:
            prompt_dict = json.load(file)
            prompt = LimerickPrompt.from_dict(prompt_dict)
            if model.in_context_window(prompt.token_count):
                question_list = prompt.question_list
                for question in question_list:
                    for location in question_location_list:
                        prompt_text = prompt.build_text_from_limerick_list(question, location, 1)
                        if prompt_text is not None:
                            prompt_text += "\n\n" + question.question
                            print("asking question at location", location)
                            result = model.prompt(prompt_text)
                            print("answered")
                            score = evaluator.evaluate(evaluator_executor, evaluator_model_list, question, result)
                            print_result(prompt, model, question, location, result, score)
                            write_prompt_text_to_file(prompt_text, prompt_file, model.llm_name, str(location), str(question.id))

def calculate_max_token_count(model_list):
    result = 0
    for model in model_list:
        if model.max_input > result:
            result = model.max_input
    return result

def create_test_directory(test_parent_directory):
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    full_path = os.path.join(test_parent_directory, date_time_str)
    os.makedirs(full_path, exist_ok=True)
    return full_path

if __name__ == '__main__':
    test_config = DEFAULT_TEST_CONFIG
    max_prompt_size = calculate_max_token_count(test_config.model_list)
    test_directory = create_test_directory(test_config.result_directory)
    run_tests(GPT3_5, ["prompt_6000.json"], ROUGH_QUESTION_LOCATIONS,
                   EVALUATOR_EXECUTOR, EVALUATOR_MODEL_LIST, DefaultEvaluator())
    print("Tests completed")
