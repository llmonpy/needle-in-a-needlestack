import concurrent
import os
from datetime import datetime

from evaluator import DefaultEvaluator
from llm_client import backoff_after_exception, PROMPT_RETRIES
from prompt import get_prompt
from test_config import DEFAULT_TEST_CONFIG, CURRENT_TEST_CONFIG
from test_results import TestResults, NO_GENERATED_ANSWER


if __name__ == '__main__':
    test_results = TestResults(CURRENT_TEST_CONFIG)
    futures_list = test_results.start()
    for future in concurrent.futures.as_completed(futures_list):
        generated_answer, score = future.result()
        print("score: ", score)
    print("Tests completed")
