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
import copy
import json
import os
import threading
from datetime import datetime
from string import Template

from evaluator import DefaultEvaluator
from limerick import Limerick, FULL_QUESTION_FILE
from llm_client import PROMPT_RETRIES, backoff_after_exception
from main import NO_GENERATED_ANSWER
from test_config import DEFAULT_TEST_CONFIG
from test_results import SYSTEM_PROMPT
from test_status import TestStatus

VETTER_STATUS_REPORT_INTERVAL = 5

QUESTION_PROMPT = """
This is a limerick:

$limerick_text

This is a question to test your understanding of the limerick:

$question_text

Please answer the question as concisely as possible. Do not explain your answer.

"""


class VetterEvaluatorResult:
    def __init__(self, model_name, passed=None):
        self.model_name = model_name
        self.passed = passed

    def set_passed(self, passed):
        self.passed = passed

    def is_finished(self):
        return self.passed is not None

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = VetterEvaluatorResult(**dictionary)
        return result


class VetterTrialResult:
    def __init__(self, trial_number, good_answer, passed=None, dissent_count=None,
                 generated_answer=None, evaluator_results=None):
        self.trial_number = trial_number
        self.good_answer = good_answer
        self.passed = passed
        self.dissent_count = dissent_count
        self.generated_answer = generated_answer
        self.evaluator_results = evaluator_results if evaluator_results else []

    def add_evaluator_result(self, model_name):
        evaluator_result = VetterEvaluatorResult(model_name)
        self.evaluator_results.append(evaluator_result)

    def is_finished(self):
        return self.passed is not None

    def has_answer(self):
        return self.generated_answer is not None

    def has_dissent(self):
        return self.dissent_count > 0

    def has_concerning_dissent(self):
        concerning_dissent_count = round(len(self.evaluator_results) / 2)
        result = self.dissent_count >= concerning_dissent_count
        return result

    def run_test(self, results, model, question, question_prompt_text, evaluator):
        for attempt in range(PROMPT_RETRIES):
            try:
                self.generated_answer = model.prompt(question_prompt_text, SYSTEM_PROMPT)
                break
            except Exception as e:
                self.generated_answer = NO_GENERATED_ANSWER
                results.test_status.add_test_exception(model.model_name, e)
                if attempt == 2:
                    results.test_status.add_answer_generation_failure(model.model_name)
                    print("Exception on attempt 3")
                backoff_after_exception(attempt)
                continue
        results.test_status.record_test_answer_generated(model.model_name)
        self.passed, evaluator_model_result_list = evaluator.evaluate(model.model_name, question, self.generated_answer)
        for evaluator_model_result in evaluator_model_result_list:
            self.set_evaluator_result(evaluator_model_result.model_name, evaluator_model_result.passed)
        results.test_status.record_test_finished(model.model_name)
        return self.generated_answer, self.passed

    def calculate_scores(self):
        finished = True
        passed_count = 0
        for evaluator_result in self.evaluator_results:
            if evaluator_result.is_finished():
                if evaluator_result.passed:
                    passed_count += 1
            else:
                finished = False
        if finished:
            self.passed = passed_count > (len(self.evaluator_results) / 2)
            self.dissent_count = 0
            for evaluator_result in self.evaluator_results:
                if evaluator_result.passed != self.passed:
                    self.dissent_count += 1

    def set_generated_answer(self, generated_answer):
        self.generated_answer = generated_answer

    def set_evaluator_result(self, evaluator_model_name, passed):
        for evaluator_result in self.evaluator_results:
            if evaluator_result.model_name == evaluator_model_name and not evaluator_result.is_finished():
                evaluator_result.set_passed(passed)
                break

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.evaluator_results is not None:
            index = 0
            for evaluator_result in self.evaluator_results:
                result["evaluator_results"][index] = evaluator_result.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        evaluator_results = dictionary.get("evaluator_results", None)
        if evaluator_results is not None:
            dictionary.pop("evaluator_results", None)
            evaluator_results = [VetterEvaluatorResult.from_dict(result) for result in evaluator_results]
            dictionary["evaluator_results"] = evaluator_results
        result = VetterTrialResult(**dictionary)
        return result

    @staticmethod
    def create(trial_number, good_answer, evaluator_model_names):
        result = VetterTrialResult(trial_number, good_answer)
        for model_name in evaluator_model_names:
            result.add_evaluator_result(model_name)
        return result


class ModelQuestionVetterResult:
    def __init__(self, model_name, question, trails=None):
        self.model_name = model_name
        self.question = question
        self.trails = trails if trails else []

    def add_trial(self, trial_number, evaluator_model_names):
        trial = VetterTrialResult.create(trial_number, self.question.answer, evaluator_model_names)
        self.trails.append(trial)

    def start_tests(self, results, model, question, question_prompt_text, evaluator):
        futures_list = []
        executor = model.get_eval_executor()
        for trial in self.trails:
            futures_list.append(executor.submit(trial.run_test, results, model, question, question_prompt_text,
                                                evaluator))
        return futures_list

    def calculate_scores(self):
        for trail in self.trails:
            trail.calculate_scores()

    def passed_tests(self):
        passed = True
        for trail in self.trails:
            if not trail.passed:
                passed = False
                break
        return passed

    def to_dict(self):
        result = copy.copy(vars(self))
        result["question"] = self.question.to_dict()
        if self.trails is not None:
            index = 0
            for trail in self.trails:
                result["trails"][index] = trail.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        trails = dictionary.get("trails", None)
        if trails is not None:
            dictionary.pop("trails", None)
            trails = [VetterTrialResult.from_dict(trail) for trail in trails]
            dictionary["trails"] = trails
        result = ModelQuestionVetterResult(**dictionary)
        return result

    @staticmethod
    def create(model_name, question, number_of_trials, evaluator_model_names):
        result = ModelQuestionVetterResult(model_name, question)
        for trial_number in range(number_of_trials):
            result.add_trial(trial_number, evaluator_model_names)
        return result


class QuestionVetterResult:
    def __init__(self, question, model_question_list=None):
        self.question = question
        self.model_question_list = model_question_list if model_question_list else []
        self.question_prompt_text = None
        self.failed_models = []

    def add_model_question(self, model_name, question, number_of_trials, evaluator_model_names):
        model_question = ModelQuestionVetterResult.create(model_name, question, number_of_trials, evaluator_model_names)
        self.model_question_list.append(model_question)

    def start_tests(self, result, model_list, evaluator):
        futures_list = []
        question_prompt_template = Template(QUESTION_PROMPT)
        self.question_prompt_text = question_prompt_template.substitute(limerick_text=self.question.text,
                                                                   question_text=self.question.question)
        for model_question in self.model_question_list:
            model_name = model_question.model_name
            model = next(model for model in model_list if model.model_name == model_name)
            if model is None:
                raise Exception("Model not found: " + model_name)
            futures_list += model_question.start_tests(result, model, self.question, self.question_prompt_text, evaluator)
        return futures_list

    def calculate_scores(self):
        for model_question in self.model_question_list:
            model_question.calculate_scores()

    def record_results(self):
        self.failed_models = []
        passed = True
        for model_question in self.model_question_list:
            if not model_question.passed_tests():
                passed = False
                self.failed_models.append(model_question.model_name)
                print("Question failed: ", self.question.id, self.question.text)
        self.question.question_vetted = passed
        return passed

    def to_dict(self):
        result = copy.copy(vars(self))
        result["question"] = self.question.to_dict()
        if self.model_question_list is not None:
            index = 0
            for model_question in self.model_question_list:
                result["model_question_list"][index] = model_question.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        model_question_list = dictionary.get("model_question_list", None)
        if model_question_list is not None:
            dictionary.pop("model_question_list", None)
            model_question_list = [ModelQuestionVetterResult.from_dict(model_question) for model_question in
                                   model_question_list]
            dictionary["model_question_list"] = model_question_list
        result = QuestionVetterResult(**dictionary)
        return result

    @staticmethod
    def create(model_name_list, question, number_of_trials, evaluator_model_names):
        result = QuestionVetterResult(question)
        for model_name in model_name_list:
            result.add_model_question(model_name, question, number_of_trials, evaluator_model_names)
        return result


class QuestionListVetterResult:
    def __init__(self, file_path, question_list=None, test_exception_list=None, failed_test_count=0,
               evaluation_exception_list=None, failed_evaluation_count=0):
        self.file_path = file_path
        self.question_list = question_list if question_list else []
        self.test_exception_list = test_exception_list
        if self.test_exception_list is None:
            self.test_exception_list = []
        self.failed_test_count = failed_test_count
        self.evaluation_exception_list = evaluation_exception_list
        if self.evaluation_exception_list is None:
            self.evaluation_exception_list = []
        self.failed_evaluation_count = failed_evaluation_count
        self.failed_questions = []

    def get_trial(self, question_id,  model_name, trial_number):
        for question_vetter_result in self.question_list:
            if question_vetter_result.question.id == question_id:
                for model_question in question_vetter_result.model_question_list:
                    if model_question.model_name == model_name:
                        for trial in model_question.trails:
                            if trial.trial_number == trial_number:
                                return trial
        return None

    def start_tests(self, results, model_list, evaluator):
        futures_list = []
        for question_vetter_result in self.question_list:
            futures_list += question_vetter_result.start_tests(results, model_list, evaluator)
        return futures_list

    def add_question(self, question, model_name_list, number_of_trials, evaluator_model_names):
        question_vetter_result = QuestionVetterResult.create(model_name_list, question, number_of_trials,
                                                             evaluator_model_names)
        self.question_list.append(question_vetter_result)

    def calculate_scores(self):
        for question_vetter_result in self.question_list:
            question_vetter_result.calculate_scores()

    def record_results(self):
        self.failed_questions = []
        for question_vetter_result in self.question_list:
            if not question_vetter_result.record_results():
                self.failed_questions.append(question_vetter_result.question.id)

    def write_to_file(self):
        with open(self.file_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.question_list is not None:
            index = 0
            for question_vetter_result in self.question_list:
                result["question_list"][index] = question_vetter_result.to_dict()
                index += 1
        return result

    def update_test_status(self, test_status):
        for question_vetter_result in self.question_list:
            for model_question in question_vetter_result.model_question_list:
                for trial in model_question.trails:
                    test_status.add_test(model_question.model_name)
                    for evaluator_result in trial.evaluator_results:
                        test_status.add_evaluation(evaluator_result.model_name)

    @staticmethod
    def from_dict(dictionary):
        question_list = dictionary.get("question_list", None)
        if question_list is not None:
            dictionary.pop("question_list", None)
            question_list = [QuestionVetterResult.from_dict(question_vetter_result) for question_vetter_result in
                             question_list]
            dictionary["question_list"] = question_list
        result = QuestionListVetterResult(**dictionary)
        return result

    @staticmethod
    def create(file_path, question_list, model_list, number_of_trials, evaluator_model_list):
        result = QuestionListVetterResult(file_path)
        model_name_list = [model.model_name for model in model_list]
        evaluator_model_names = [model.model_name for model in evaluator_model_list]
        for question in question_list:
            result.add_question(question, model_name_list, number_of_trials, evaluator_model_names)
        return result


class QuestionListVetter:
    def __init__(self, directory, question_list, model_list, number_of_trials, evaluator_model_list):
        self.directory = directory
        self.question_list = question_list
        self.model_list = model_list
        self.number_of_trials = number_of_trials
        self.evaluator_model_list = evaluator_model_list
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        full_path = os.path.join(directory, date_time_str + ".json")
        self.test_status = TestStatus(model_list, evaluator_model_list)
        self.result = QuestionListVetterResult.create(full_path, question_list, model_list, number_of_trials,
                                                      evaluator_model_list)
        self.result.update_test_status(self.test_status)

    def start(self):
        evaluator = DefaultEvaluator(self.test_status, self.evaluator_model_list)
        futures_list = self.result.start_tests(self, self.model_list, evaluator)
        self.test_status.start(self)
        return futures_list

    def all_tests_finished(self):
        self.result.calculate_scores()
        self.result.record_results()
        self.result.write_to_file()
        print("All tests are finished")
        exit(0)

    @staticmethod
    def from_file(question_file_path, result_directory, model_list, number_of_trials, evaluator_model_list):
        with open(question_file_path, "r") as file:
            question_dict_list = json.load(file)
        question_list = [Limerick.from_dict(question_dict) for question_dict in question_dict_list]
        os.makedirs(result_directory, exist_ok=True)
        result = QuestionListVetter(result_directory, question_list, model_list, number_of_trials, evaluator_model_list)
        return result


if __name__ == '__main__':
    test_config = DEFAULT_TEST_CONFIG
    vetter = QuestionListVetter.from_file(FULL_QUESTION_FILE, "vetter_results",
                                          test_config.model_list, 5,
                                          test_config.evaluator_model_list)
    vetter_futures_list = vetter.start()
    for future in concurrent.futures.as_completed(vetter_futures_list):
        generated_answer, score = future.result()
    print("Tests completed")


