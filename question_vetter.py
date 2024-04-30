import concurrent
import copy
import json
import os
import threading
from datetime import datetime
from string import Template

from base_test_results import BaseStatusReport, BaseTestResults
from evaluator import DefaultEvaluator
from limerick import Limerick, FULL_QUESTION_FILE
from llm_client import PROMPT_RETRIES, backoff_after_exception
from main import NO_GENERATED_ANSWER
from test_config import DEFAULT_TEST_CONFIG

VETTER_STATUS_REPORT_INTERVAL = 5

QUESTION_PROMPT = """
This is a limerick:

$limerick_text

This is a question to test your understanding of the limerick:

$question_text

Please answer the question as concisely as possible. Do not explain your answer.

"""


class VetterTestResultExceptionReport:
    def __init__(self, model_name, question_id, trial_number, attempt, exception_message):
        self.model_name = model_name
        self.question_id = question_id
        self.trial_number = trial_number
        self.attempt = attempt
        self.exception_message = exception_message

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = VetterTestResultExceptionReport(**dictionary)
        return result


class VetterEvaluationExceptionReport:
    def __init__(self, model_name, question_id, trial_number, attempt, evaluation_model_name, exception_message):
        self.model_name = model_name
        self.question_id = question_id
        self.trial_number = trial_number
        self.attempt = attempt
        self.evaluation_model_name = evaluation_model_name
        self.exception_message = exception_message

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = VetterEvaluationExceptionReport(**dictionary)
        return result


class VetterStatusReport(BaseStatusReport):
    def __init__(self, failed_test_count=0, failed_evaluation_count=0):
        super().__init__(failed_test_count, failed_evaluation_count)

    def print(self):
        print("Tests: ", self.test_count, " Answered: ", self.answered_test_count, " Finished: ",
              self.finished_test_count)
        print("Evaluators: ", self.evaluator_count, " Finished: ", self.finished_evaluator_count)
        for evaluator_model_name in self.waiting_for_evaluator_count:
            count = self.waiting_for_evaluator_count[evaluator_model_name]
            print("Waiting for evaluator: ", evaluator_model_name, " Count: ", count)
        print("Failed Tests: ", self.failed_test_count, " Failed Evaluations: ", self.failed_evaluation_count)
        print("--------------------")


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

    '''
        You might wonder why this doesn't just set the generated_answer directly, rather than going
        through results.set_test_result.  The reasons are: 1) to keep the code consistent with the
        way the evaluator results are set, and 2) need "results" to record exceptions and 3) generate_answer
        would need a lock to safely set the value (because it is accessed by the update status thread). It's
        still weird though.
    '''
    def run_test(self, results, model, question, question_prompt_text, evaluator):
        for attempt in range(PROMPT_RETRIES):
            try:
                generated_answer = model.prompt(question_prompt_text)
                break
            except Exception as e:
                generated_answer = NO_GENERATED_ANSWER
                results.add_test_exception(model.llm_name, None, question.id, self.trial_number, attempt, e)
                if attempt == 2:
                    print("Exception on attempt 3")
                backoff_after_exception(attempt)
                continue
        results.set_test_result(model.llm_name, None, question.id, self.trial_number, generated_answer)
        score = evaluator.evaluate(model.llm_name, None, question, self.trial_number, generated_answer)
        return generated_answer, score

    def calculate_scores(self, status_report):
        finished = True
        passed_count = 0
        for evaluator_result in self.evaluator_results:
            status_report.add_evaluator_test(evaluator_result.is_finished(), evaluator_result.model_name)
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
        status_report.add_test(self.has_answer(), self.is_finished())

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

    def calculate_scores(self, status_report):
        for trail in self.trails:
            trail.calculate_scores(status_report)

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

    def add_model_question(self, model_name, question, number_of_trials, evaluator_model_names):
        model_question = ModelQuestionVetterResult.create(model_name, question, number_of_trials, evaluator_model_names)
        self.model_question_list.append(model_question)

    def start_tests(self, result, model_list, evaluator):
        futures_list = []
        question_prompt_template = Template(QUESTION_PROMPT)
        question_prompt_text = question_prompt_template.substitute(limerick_text=self.question.text,
                                                                   question_text=self.question.question)
        for model_question in self.model_question_list:
            model_name = model_question.model_name
            model = next(model for model in model_list if model.llm_name == model_name)
            if model is None:
                raise Exception("Model not found: " + model_name)
            futures_list += model_question.start_tests(result, model, self.question, question_prompt_text, evaluator)
        return futures_list

    def calculate_scores(self, status_report):
        for model_question in self.model_question_list:
            model_question.calculate_scores(status_report)

    def record_results(self):
        passed = True
        for model_question in self.model_question_list:
            if not model_question.passed_tests():
                passed = False
                print("Question failed: ", self.question.id, self.question.text)
                break
        self.question.question_vetted = passed

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

    def add_test_exception(self, model_name, question_id, trial_number, attempt, exception):
        print("Test Exception: ", str(exception))
        exception_report = VetterTestResultExceptionReport(model_name, question_id, trial_number, attempt,
                                                           str(exception))
        self.test_exception_list.append(exception_report)
        if attempt == PROMPT_RETRIES - 1:
            self.failed_test_count += 1
            print("Failed Test Count: ", self.failed_test_count)

    def add_evaluation_exception(self, model_name, question_id, trial_number, attempt, evaluation_model_name,
                                 exception):
        print("Evaluation Exception: ", str(exception))
        exception_report = VetterEvaluationExceptionReport(model_name, question_id, trial_number, attempt,
                                                           evaluation_model_name, str(exception))
        self.evaluation_exception_list.append(exception_report)
        if attempt == PROMPT_RETRIES - 1:
            self.failed_evaluation_count += 1
            print("Failed Evaluation Count: ", self.failed_evaluation_count)

    def add_question(self, question, model_name_list, number_of_trials, evaluator_model_names):
        question_vetter_result = QuestionVetterResult.create(model_name_list, question, number_of_trials,
                                                             evaluator_model_names)
        self.question_list.append(question_vetter_result)

    def calculate_scores(self, status_report):
        for question_vetter_result in self.question_list:
            question_vetter_result.calculate_scores(status_report)

    def record_results(self):
        for question_vetter_result in self.question_list:
            question_vetter_result.record_results()

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
        if self.test_exception_list is not None:
            index = 0
            for test_exception in self.test_exception_list:
                result["test_exception_list"][index] = test_exception.to_dict()
                index += 1
        if self.evaluation_exception_list is not None:
            index = 0
            for evaluation_exception in self.evaluation_exception_list:
                result["evaluation_exception_list"][index] = evaluation_exception.to_dict()
                index += 1
        return result

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
        model_name_list = [model.llm_name for model in model_list]
        evaluator_model_names = [model.llm_name for model in evaluator_model_list]
        for question in question_list:
            result.add_question(question, model_name_list, number_of_trials, evaluator_model_names)
        return result


class BaseVetterTestResultAction:
    def __init__(self, results, model_name, question_id, trial_number):
        self.results = results
        self.model_name = model_name
        self.question_id = question_id
        self.trial_number = trial_number

    def get_trial(self):
        result = self.results.get_trial(self.question_id, self.model_name, self.trial_number)
        return result

    def execute(self):
        raise NotImplementedError


class SetVetterTestResultAction(BaseVetterTestResultAction):
    def __init__(self, results, model_name, question_id, trial_number, generated_answer):
        super().__init__(results, model_name, question_id, trial_number)
        self.generated_answer = generated_answer

    def execute(self):
        self.get_trial().set_generated_answer(self.generated_answer)


class SetVetterEvaluatorResultAction(BaseVetterTestResultAction):
    def __init__(self, results, model_name, question_id, trial_number, evaluator_model_name, passed):
        super().__init__(results, model_name, question_id, trial_number)
        self.evaluator_model_name = evaluator_model_name
        self.passed = passed

    def execute(self):
        self.get_trial().set_evaluator_result(self.evaluator_model_name, self.passed)


class AddVetterTestExceptionAction(BaseVetterTestResultAction):
    def __init__(self, results, model_name, question_id, trial_number, attempt, exception):
        super().__init__(results, model_name, question_id, trial_number)
        self.attempt = attempt
        self.exception = exception

    def execute(self):
        self.results.add_test_exception(self.model_name, self.question_id, self.trial_number, self.attempt,
                                        self.exception)


class AddVetterEvaluationExceptionAction(BaseVetterTestResultAction):
    def __init__(self, results, model_name, question_id, trial_number, evaluation_model_name, attempt,
                 exception):
        super().__init__(results, model_name, question_id, trial_number)
        self.evaluation_model_name = evaluation_model_name
        self.attempt = attempt
        self.exception = exception

    def execute(self):
        self.results.add_evaluation_exception(self.model_name, self.question_id, self.trial_number,
                                              self.evaluation_model_name, self.attempt, self.exception)


class QuestionListVetter(BaseTestResults):
    def __init__(self, directory, question_list, model_list, number_of_trials, evaluator_model_list):
        self.directory = directory
        self.question_list = question_list
        self.model_list = model_list
        self.number_of_trials = number_of_trials
        self.evaluator_model_list = evaluator_model_list
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        full_path = os.path.join(directory, date_time_str + ".json")
        self.result = QuestionListVetterResult.create(full_path, question_list, model_list, number_of_trials,
                                                      evaluator_model_list)
        self.action_list = []
        self.action_list_lock = threading.Lock()
        self.started = False
        self.timer = None

    def add_action(self, action):
        with self.action_list_lock:
            self.action_list.append(action)

    def reset_and_return_actions(self):
        with self.action_list_lock:
            result = self.action_list
            self.action_list = []
        return result

    def execute_actions(self):
        actions = self.reset_and_return_actions()
        for action in actions:
            try:
                action.execute()
            except Exception as e:
                print("Exception executing action: ", str(e))
                pass

    def start(self):
        evaluator = DefaultEvaluator(self, self.evaluator_model_list)
        futures_list = self.result.start_tests(self, self.model_list, evaluator)
        self.started = True
        self.timer = threading.Timer(interval=VETTER_STATUS_REPORT_INTERVAL, function=self.update_and_report_status)
        self.timer.start()
        return futures_list

    def set_test_result(self, model_name, location_name, question_id, trial_number, generated_answer):
        action = SetVetterTestResultAction(self.result, model_name, question_id, trial_number, generated_answer)
        self.add_action(action)

    def set_evaluator_result(self, model_name, location_name, question_id, trial_number, evaluator_model_name, passed):
        action = SetVetterEvaluatorResultAction(self.result, model_name, question_id, trial_number,
                                                evaluator_model_name, passed)
        self.add_action(action)

    def add_test_exception(self, model_name, location_name, question_id, trial_number, attempt, exception):
        action = AddVetterTestExceptionAction(self.result, model_name, question_id, trial_number, attempt, exception)
        self.add_action(action)

    def add_evaluation_exception(self, model_name, location_name, question_id, trial_number, evaluation_model_name,
                                 attempt, exception):
        action = AddVetterEvaluationExceptionAction(self.result, model_name, question_id, trial_number,
                                                    evaluation_model_name, attempt, exception)
        self.add_action(action)

    def update_and_report_status(self):
        if not self.started:
            return
        self.execute_actions()
        status_report = VetterStatusReport()
        self.result.calculate_scores(status_report)
        if not status_report.is_finished():
            status_report.print()
            self.timer = threading.Timer(interval=VETTER_STATUS_REPORT_INTERVAL, function=self.update_and_report_status)
            self.timer.start()
        else:
            self.result.record_results()
            self.result.write_to_file()
            print("All tests are finished")

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
        print("score: ", score)
    print("Tests completed")
