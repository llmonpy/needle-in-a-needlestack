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
        score = 0  # means the LLM did not give answer the question as asked
    elif is_pass:
        score = 1
    elif is_fail:
        score = 0
    return score


def evaluate_response(model, evaluation_prompt_text, system_prompt, model_name_being_tested, location_name, question,
                      cycle_number, results):
    for attempt in range(PROMPT_RETRIES):
        try:
            response_text = model.prompt(evaluation_prompt_text, system_prompt)
            break
        except Exception as e:
            response_text = FAIL_ANSWER
            results.add_evaluation_exception(model_name_being_tested, location_name, question.id, cycle_number,
                                             model.llm_name, attempt, e)
            if attempt == 2:
                print("Exception on attempt 3")
            backoff_after_exception(attempt)
            continue
    score = get_score_from_response(response_text)
    return score, model.llm_name


class EvaluatorInterface:
    def evaluate(self, results, evaluator_model_list, model_name, location_name, question, cycle_number, answer):
        raise NotImplementedError


class DefaultEvaluator(EvaluatorInterface):
    def __init__(self, results, evaluator_model_list, evaluation_prompt=EVALUATION_PROMPT, system_prompt=SYSTEM_PROMPT):
        self.evaluation_prompt = evaluation_prompt
        self.system_prompt = system_prompt
        self.results = results
        self.evaluator_model_list = evaluator_model_list

    def evaluate(self, model_name, location_name, question, cycle_number, answer):
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
                                                model_name, location_name, question, cycle_number, self.results))
        yes_count = no_count = 0
        for future in concurrent.futures.as_completed(futures_list):
            score, evaluator_model_name = future.result()
            if score == 1:
                yes_count += 1
                passed = True
            elif score == 0:
                no_count += 1
                passed = False
            self.results.set_evaluator_result(model_name, location_name, question.id, cycle_number, evaluator_model_name,
                                         passed)
        if yes_count > no_count:
            result = 1
        else:
            result = 0
        return result


