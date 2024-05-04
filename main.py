import concurrent
import os
from datetime import datetime

from evaluator import DefaultEvaluator
from llm_client import backoff_after_exception, PROMPT_RETRIES
from prompt import get_prompt
from test_config import DEFAULT_TEST_CONFIG, CURRENT_TEST_CONFIG
from test_results import TestResults, NO_GENERATED_ANSWER


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


def test_model(results, prompt_text, model, evaluator, location, question, trial_number):
    for attempt in range(PROMPT_RETRIES):
        try:
            generated_answer = model.prompt(prompt_text)
            break
        except Exception as e:
            generated_answer = NO_GENERATED_ANSWER
            results.add_test_exception(model.llm_name, location, question.id, trial_number, attempt, e)
            if attempt == 2:
                print("Exception on attempt 3")
            backoff_after_exception(attempt)
            continue
    results.set_test_result(model.llm_name, location, question.id, trial_number, generated_answer)
    score = evaluator.evaluate(model.llm_name, location, question, trial_number, generated_answer)
    return generated_answer, score


def run_tests_for_model(results, prompt, model, config, evaluator):
    print("starting tests for model: ", model.llm_name)
    futures_list = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.test_thread_count)
    question_list = prompt.question_list
    question_location_list = calculate_question_location_list(config.location_count, model.max_input)
    model_results = results.add_model(model.llm_name, question_location_list, question_list, config.trials, config.evaluator_model_list)
    for question in question_list:
        for location in question_location_list:
            prompt_text = prompt.build_text_from_limerick_list(question, location, model.max_input,
                                                               config.repeat_question_limerick_count)
            prompt_text += "\n\n" + question.question
            write_prompt_text_to_file(prompt_text, model_results, config, str(location), str(question.id))
            for trial_number in range(config.trials):
                futures_list.append(executor.submit(test_model, results, prompt_text, model, evaluator,
                                                    location, question, trial_number))
    print("Number of tests ", len(futures_list))
    return futures_list


def run_tests(results, prompt, config, evaluator):
    futures_list = []
    for model in config.model_list:
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
    test_results = TestResults(CURRENT_TEST_CONFIG)
    futures_list = test_results.start()
    for future in concurrent.futures.as_completed(futures_list):
        generated_answer, score = future.result()
        print("score: ", score)
    print("Tests completed")
