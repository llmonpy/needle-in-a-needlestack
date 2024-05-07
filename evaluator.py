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
import concurrent
from string import Template

from llm_client import backoff_after_exception
from llm_client import PROMPT_RETRIES

# Using meaningless strings to avoid confusion.  Sometimes the correct answer to a question about a limerick is "No"
# and the LLMs sometimes got confused and said that "No" was the incorrect answer when the correct answer was "No".
PASS_ANSWER = "aaa"
FAIL_ANSWER = "bbb"

SYSTEM_PROMPT = Template("""
You are an expert at evaluating the answers to questions based on the text of a limerick. You are
sure of yourself and always answer with an "$pass_answer" or "$fail_answer" without explanation.""").substitute(pass_answer=PASS_ANSWER,
                                                                                               fail_answer=FAIL_ANSWER)

EVALUATION_PROMPT = """
I would like you to evaluate the answer to a question about a limerick.  The limerick is:

$limerick_text

The question is:

$question_text

An example of a good answer to this question is:

$good_answer_text

And this is the generated answer to the question:

$generated_answer_text

You need to determine if the generated answer passes or fails.  Pass means the answer has the same meaning as the good 
answer.  It does not matter if the generated answer is more or less concise than the good answer.  Does the generated
answer pass or fail? Please answer with an "aaa" if the generated answer passes or "bbb" if it fails. Do not provide 
an explanation, only reply with "$pass_answer" or "$fail_answer".

"""


def get_score_from_response(response_text):
    score = None
    is_pass = is_fail = False
    if response_text is not None and len(response_text) > 0:
        response_text = response_text.lower()
        is_pass = PASS_ANSWER in response_text
        is_fail = FAIL_ANSWER in response_text
    if is_pass == is_fail:
        score = 0  # means the LLM did not answer the question as asked
    elif is_pass:
        score = 1
    elif is_fail:
        score = 0
    return score


def evaluate_response(model, evaluation_prompt_text, system_prompt, model_name_being_tested, test_status):
    for attempt in range(PROMPT_RETRIES):
        try:
            response_text = model.prompt(evaluation_prompt_text, system_prompt)
            break
        except Exception as e:
            response_text = FAIL_ANSWER
            test_status.add_evaluation_exception(model.model_name, e)
            if attempt == 2:
                test_status.add_evaluation_failure(model_name_being_tested, model.model_name)
                print("Exception on attempt 3")
            else:
                backoff_after_exception(attempt)
            continue
    score = get_score_from_response(response_text)
    return score, model.model_name


class EvaluatorInterface:
    def evaluate(self, model_name, question, answer):
        raise NotImplementedError


class EvaluatorResult:
    def __init__(self, model_name, passed):
        self.model_name = model_name
        self.passed = passed


class DefaultEvaluator(EvaluatorInterface):
    def __init__(self, test_status, evaluator_model_list, evaluation_prompt=EVALUATION_PROMPT, system_prompt=SYSTEM_PROMPT):
        self.evaluation_prompt = evaluation_prompt
        self.system_prompt = system_prompt
        self.test_status = test_status
        self.evaluator_model_list = evaluator_model_list

    def evaluate(self, model_name, question, answer):
        model_results = []
        evaluation_prompt_template = Template(self.evaluation_prompt)
        evaluation_prompt_text = evaluation_prompt_template.substitute(limerick_text=question.text,
                                                                       question_text=question.question,
                                                                       good_answer_text=question.answer,
                                                                       generated_answer_text=answer,
                                                                       pass_answer=PASS_ANSWER,
                                                                       fail_answer=FAIL_ANSWER)
        futures_list = []
        for model in self.evaluator_model_list:
            executor = model.get_eval_executor()
            futures_list.append(executor.submit(evaluate_response, model, evaluation_prompt_text, self.system_prompt,
                                                model_name, self.test_status))
        yes_count = no_count = 0
        for future in concurrent.futures.as_completed(futures_list):
            score, evaluator_model_name = future.result()
            if score == 1:
                yes_count += 1
                passed = True
            else:
                no_count += 1
                passed = False
            model_results.append(EvaluatorResult(evaluator_model_name, passed))
            self.test_status.record_evaluation_finished(evaluator_model_name)
        if yes_count > no_count:
            final_result = True
        else:
            final_result = False
        return final_result, model_results


