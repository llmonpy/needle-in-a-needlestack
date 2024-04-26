import concurrent
import os
from datetime import datetime

from evaluator import DefaultEvaluator
from prompt import get_prompt
from test_config import DEFAULT_TEST_CONFIG
from test_results import TestResults


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


def write_prompt_text_to_file(prompt_text, model_results, config, location, question_id):
    if config.write_prompt_text_to_file:
        file_name = "p_" + location + "_" + question_id + ".txt"
        file_path = os.path.join(model_results.directory, file_name)
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


def test_model(results, prompt_text, model, config, evaluator, location, question, cycle_number):
    try:
        generated_answer = model.prompt(prompt_text)
    except Exception as e:
        raise e
    results.set_test_result(model.llm_name, location, question.id, cycle_number, generated_answer)
    score = evaluator.evaluate(results, config.evaluator_model_list, model.llm_name, location, question, cycle_number,
                               generated_answer)
    return generated_answer, score


def run_tests_for_model(results, prompt, model, config, evaluator):
    futures_list = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.test_thread_count)
    question_list = prompt.question_list
    question_location_list = calculate_question_location_list(config.location_count, model.max_input)
    model_results = results.add_model(model.llm_name, question_location_list, question_list, config.cycles, config.evaluator_model_list)
    for question in question_list:
        for location in question_location_list:
            prompt_text = prompt.build_text_from_limerick_list(question, location, 1)
            prompt_text += "\n\n" + question.question
            write_prompt_text_to_file(prompt_text, model_results, config, str(location), str(question.id))
            for cycle_number in range(config.cycles):
                futures_list.append(executor.submit(test_model, results, prompt_text, model, config, evaluator,
                                                    location, question, cycle_number))
    return futures_list


def run_tests(results, prompt, config, evaluator):
    futures_list = []
    for model in test_config.model_list:
        futures_list += run_tests_for_model(results, prompt, model, config, evaluator)
    results.start()
    for future in concurrent.futures.as_completed(futures_list):
        generated_answer, score = future.result()
        print("score: ", score)


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
    test_results = TestResults(test_directory)
    run_tests(test_results, test_prompt, test_config, DefaultEvaluator())
    print("Tests completed")
