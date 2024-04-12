from string import Template

from llm_client import ANTHROPIC_HAIKU, GPT3_5, ANTHROPIC_SONNET, GPT4

SYSTEM_PROMPT = """
You are an expert at analyzing the answers to questions based on the text of a limerick. You are
sure of yourself and always answer with an "aaa" or "bbb" without explanation."""

EVALUATION_PROMPT = """
I would like you to evaluate the answer to a question about a limerick.  The limerick is:

$limerick_text

The question is:

$question_text

An example of a good answer to this question is:

$good_answer_text

And this is the generated answer to the question:

$generated_answer_text

A good answer demonstrates understanding of the limerick and the question.  It does not matter if the answer is less
or more concise than the example answer.  Is the generated answer below also good answer to the question?



Please answer with an "aaa" if the generated answer is semantically similar to the good answer and a "bbb" if its not.
It does not matter generated answer is more or less concise than the good answer.  Do not provide an explanation, only
reply with "aaa" or "bbb".

"""


def get_score_from_response(response_text):
    score = None
    is_yes = is_no = False
    if response_text is not None and len(response_text) > 0:
        response_text = response_text.lower()
        is_yes = "aaa" in response_text
        is_no = "bbb" in response_text
    if is_yes == is_no:
        score = None  # means the LLM did not give answer the question as asked
    elif is_yes:
        score = 1
    elif is_no:
        score = 0
    return score


class EvaluatorInterface:
    def evaluate(self, executor, question, response):
        raise NotImplementedError


class DefaultEvaluator(EvaluatorInterface):
    def __init__(self, primary_llm=ANTHROPIC_HAIKU, secondary_llm=GPT3_5):
        self.primary_llm = primary_llm
        self.secondary_llm = secondary_llm

    def evaluate(self, executor, question, answer):
        evaluation_prompt_template = Template(EVALUATION_PROMPT)
        evaluation_prompt_text = evaluation_prompt_template.substitute(limerick_text=question.text,
                                                                       question_text=question.question,
                                                                       good_answer_text=question.answer,
                                                                       generated_answer_text=answer)
        response_text = self.primary_llm.prompt(evaluation_prompt_text, SYSTEM_PROMPT)
        print("response text: ", response_text)
        score = get_score_from_response(response_text)
        if score is None:
            print("retry evaluation with secondary LLM")
            response_text = self.secondary_llm.prompt(evaluation_prompt_text, SYSTEM_PROMPT)
            print("response text: ", response_text)
            score = get_score_from_response(response_text)
        if score is None:
            print("Could not evaluate the response.")
            score = 0 # assume it is a bad response if we cannot evaluate it
        return score
