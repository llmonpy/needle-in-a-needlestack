import concurrent
import copy
import json
import os
import random
from datetime import datetime

from evaluator import DefaultEvaluator
from limerick import Limerick, read_and_init_limericks, FULL_QUESTION_FILE
from llm_client import GPT3_5
from prompt import get_prompt
from test_config import DEFAULT_TEST_CONFIG, TEST_MODEL_LIST, EVALUATOR_MODEL_LIST

NUMBER_OF_QUESTIONS_PER_PROMPT = 5



ROUGH_QUESTION_LOCATIONS = [100, 1200, 5700]

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


def calculate_question_location_list(location_count, max_input):
    result = []
    increment = round(max_input / location_count)
    max_input = round(max_input * 0.90)
    initial_location = last_location = round(max_input * 0.01)
    result.append(initial_location)
    for i in range(1,location_count):
        result.append(last_location + increment)
        last_location = result[-1]
    return result


def test_model(prompt_text, model, config, evaluator, question):
    result = model.prompt(prompt_text)
    print("answered q " + str(question.id))
    score = evaluator.evaluate(config.evaluator_model_list, question, result)
    return result, score

def run_tests_for_model(prompt, model, config, evaluator):
    futures_list = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.test_thread_count)
    question_list = prompt.question_list
    question_location_list = calculate_question_location_list(config.location_count, model.max_input)
    for question in question_list:
        for location in question_location_list:
            prompt_text = prompt.build_text_from_limerick_list(question, location, 1)
            prompt_text += "\n\n" + question.question
            print("asking question at location", location)
            write_prompt_text_to_file(prompt_text, config.prompt_file_name, model.llm_name, str(location), str(question.id))
            for i in range(config.cycles):
                futures_list.append(executor.submit(test_model, prompt_text, model, config, evaluator, question))
    for future in concurrent.futures.as_completed(futures_list):
        result, score = future.result()
        print("score: ", score)


def run_tests(prompt, config, evaluator):
    for model in test_config.model_list:
        run_tests_for_model(prompt, model, config, evaluator)


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
    test_prompt = get_prompt(max_prompt_size, test_config)
    run_tests( test_prompt, test_config, DefaultEvaluator())
    print("Tests completed")
