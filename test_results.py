import copy
import json
import os
import threading
import matplotlib.pyplot as plot

from limerick import Limerick
from llm_client import PROMPT_RETRIES

STATUS_REPORT_INTERVAL = 5 # seconds


class TestResultExceptionReport:
    def __init__(self, location_name, question_id, cycle_number, attempt, exception_message):
        self.location_name = location_name
        self.question_id = question_id
        self.cycle_number = cycle_number
        self.attempt = attempt
        self.exception_message = exception_message

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = TestResultExceptionReport(**dictionary)
        return result


class EvaluationExceptionReport:
    def __init__(self, location_name, question_id, cycle_number, attempt, evaluation_model_name, exception_message):
        self.location_name = location_name
        self.question_id = question_id
        self.cycle_number = cycle_number
        self.attempt = attempt
        self.evaluation_model_name = evaluation_model_name
        self.exception_message = exception_message

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = EvaluationExceptionReport(**dictionary)
        return result


class StatusReport:
    def __init__(self, model_name, failed_test_count=0, failed_evaluation_count=0):
        self.model_name = model_name
        self.failed_test_count = failed_test_count
        self.failed_evaluation_count = failed_evaluation_count
        self.test_count = 0
        self.answered_test_count = 0
        self.finished_test_count = 0
        self.evaluator_count = 0
        self.finished_evaluator_count = 0
        self.waiting_for_evaluator_count = {}

    def add_test(self, has_answer, is_finished):
        self.test_count += 1
        if has_answer:
            self.answered_test_count += 1
        if is_finished:
            self.finished_test_count += 1

    def add_evaluator_test(self, finished, evaluator_model_name):
        self.evaluator_count += 1
        if finished:
            self.finished_evaluator_count += 1
        else:
            current_count = self.waiting_for_evaluator_count.get(evaluator_model_name, 0)
            self.waiting_for_evaluator_count[evaluator_model_name] = current_count + 1

    def is_finished(self):
        result = self.test_count == self.finished_test_count
        return result

    def print(self):
        print("Model: ", self.model_name)
        print("Tests: ", self.test_count, " Answered: ", self.answered_test_count, " Finished: ", self.finished_test_count)
        print("Evaluators: ", self.evaluator_count, " Finished: ", self.finished_evaluator_count)
        for evaluator_model_name in self.waiting_for_evaluator_count:
            count = self.waiting_for_evaluator_count[evaluator_model_name]
            print("Waiting for evaluator: ", evaluator_model_name, " Count: ", count)
        print("Failed Tests: ", self.failed_test_count, " Failed Evaluations: ", self.failed_evaluation_count)
        print("--------------------")


class ScoreAccumulator:
    def __init__(self):
        self.cumulative_score = 0
        self.count = 0

    def add_score(self, score):
        self.cumulative_score += score
        self.count += 1

    def get_score(self):
        if self.count > 0:
            result = self.cumulative_score / self.count
        else:
            result = 0
        return result


class CycleScore:
    def __init__(self, cycle_number, score):
        self.cycle_number = cycle_number
        self.score = score

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = CycleScore(**dictionary)
        return result


class LocationScore:
    def __init__(self, location_token_position, score, cycle_scores=None):
        self.location_token_position = location_token_position
        self.score = score
        self.cycle_scores = cycle_scores

    def get_cycle_score(self, cycle_number):
        result = None
        for cycle_score in self.cycle_scores:
            if cycle_score.cycle_number == cycle_number:
                result = cycle_score
                break
        return result

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.cycle_scores is not None:
            index = 0
            for cycle_score in self.cycle_scores:
                result["cycle_scores"][index] = cycle_score.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        cycle_scores = dictionary.get("cycle_scores", None)
        if cycle_scores is not None:
            dictionary.pop("cycle_scores", None)
            cycle_scores = [CycleScore.from_dict(cycle) for cycle in cycle_scores]
            dictionary["cycle_scores"] = cycle_scores
        result = LocationScore(**dictionary)
        return result


class ModelScore:
    def __init__(self, model_name, location_scores=None):
        self.model_name = model_name
        self.location_scores = location_scores

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.location_scores is not None:
            index = 0
            for location_score in self.location_scores:
                result["location_scores"][index] = location_score.to_dict()
                index += 1
        return result

    def get_location_cycle_scores(self):
        cycle_name_list = [cycle_score.cycle_number for cycle_score in self.location_scores[0].cycle_scores]
        result = []
        for cycle_name in cycle_name_list:
            location_cycle_scores = []
            for location_score in self.location_scores:
                cycle_score = location_score.get_cycle_score(cycle_name)
                location_cycle_scores.append(cycle_score.score * 100)
            result.append(location_cycle_scores)
        return result

    def write_plot(self, plot_file_name):
        labels = [location.location_token_position for location in self.location_scores]
        values = [round(location.score * 100) for location in self.location_scores]
        get_location_cycle_scores = self.get_location_cycle_scores()
        plot.figure(figsize=(10, 10))
        for location_cycle_scores in get_location_cycle_scores:
            plot.plot(labels, location_cycle_scores, linewidth=1, color="#b3d0fc", alpha=0.5)
        plot.plot(labels, values, linewidth=3, color='black', label="Average", marker='o')
        plot.title('Limerick Question Answering')
        plot.xlabel('Location(tokens)')
        plot.ylabel('Percent Correct')
        plot.grid(True)
        plot.ylim(0, 100)
        plot.savefig(plot_file_name, dpi=300)

    @staticmethod
    def from_dict(dictionary):
        location_scores = dictionary.get("location_scores", None)
        if location_scores is not None:
            dictionary.pop("location_scores", None)
            location_scores = [LocationScore.from_dict(location) for location in location_scores]
            dictionary["location_scores"] = location_scores
        result = ModelScore(**dictionary)
        return result


class TestModelScores:
    def __init__(self, model_scores=None):
        self.model_scores = model_scores

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.model_scores is not None:
            index = 0
            for model_score in self.model_scores:
                result["model_scores"][index] = model_score.to_dict()
                index += 1
        return result

    def write_to_file(self, file_path):
        with open(file_path, "w") as file:
            results_dict = self.to_dict()
            json.dump(results_dict, file, indent=4)

    @staticmethod
    def from_dict(dictionary):
        model_scores = dictionary.get("model_scores", None)
        if model_scores is not None:
            dictionary.pop("model_scores", None)
            model_scores = [ModelScore.from_dict(result) for result in model_scores]
            dictionary["model_scores"] = model_scores
        result = TestModelScores(**dictionary)
        return result



class EvaluatorResult:
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
        result = EvaluatorResult(**dictionary)
        return result


class CycleResult:
    def __init__(self, cycle_number, evaluator_model_list, good_answer, passed=None, dissent_count=None,
                 generated_answer=None, evaluator_results=None):
        self.cycle_number = cycle_number
        self.good_answer = good_answer
        self.passed = None
        self.dissent_count = None
        self.generated_answer = None
        self.evaluator_results = evaluator_results
        if evaluator_results is None:
            self.evaluator_results = []
            for model in evaluator_model_list:
                evaluator_results = EvaluatorResult(model.llm_name)
                self.evaluator_results.append(evaluator_results)

    def is_finished(self):
        return self.passed is not None

    def has_answer(self):
        return self.generated_answer is not None

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
            evaluator_results = [EvaluatorResult.from_dict(result) for result in evaluator_results]
            dictionary["evaluator_results"] = evaluator_results
        result = CycleResult(**dictionary)
        return result


class QuestionResults:
    def __init__(self, question, cycles, evaluator_model_list, cycle_results=None):
        self.question = question
        self.cycle_results = cycle_results
        if cycle_results is None:
            self.cycle_results = []
            for cycle in range(cycles):
                cycle_result = CycleResult(cycle, evaluator_model_list, question.answer)
                self.cycle_results.append(cycle_result)
        self.score = None

    def get_cycle(self, cycle_number):
        result = self.cycle_results[cycle_number]
        return result

    def get_cycle_names(self):
        result = [cycle_result.cycle_number for cycle_result in self.cycle_results]
        return result

    def add_score_for_cycle(self, cycle_name, score_accumulator):
        for cycle_result in self.cycle_results:
            if cycle_result.cycle_number == cycle_name:
                score_accumulator.add_score(cycle_result.passed)
                break

    def calculate_scores(self, status_report):
        correct_results = 0
        finished_cycles = 0
        for cycle_result in self.cycle_results:
            cycle_result.calculate_scores(status_report)
            status_report.add_test(cycle_result.has_answer(), cycle_result.is_finished())
            if cycle_result.is_finished():
                finished_cycles += 1
                if cycle_result.passed:
                    correct_results += 1
        if finished_cycles > 0:
            self.score = correct_results / finished_cycles
        return self.score

    def to_dict(self):
        result = copy.copy(vars(self))
        result["question"] = self.question.to_dict()
        if self.cycle_results is not None:
            index = 0
            for cycle_result in self.cycle_results:
                result["cycle_results"][index] = cycle_result.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        question = dictionary.get("question", None)
        if question is not None:
            dictionary.pop("question", None)
            question = Limerick.from_dict(question)
            dictionary["question"] = question
        cycle_results = dictionary.get("cycle_results", None)
        if cycle_results is not None:
            dictionary.pop("cycle_results", None)
            cycle_results = [CycleResult.from_dict(result) for result in cycle_results]
            dictionary["cycle_results"] = cycle_results
        result = QuestionResults(**dictionary)
        return result


class LocationResults:
    def __init__(self, location_token_position, question_list, cycles, evaluator_model_list, question_result_list=None):
        self.location_token_position = location_token_position
        self.question_result_list = question_result_list
        if self.question_result_list is None:
            self.question_result_list = []
            for question in question_list:
                question_result = QuestionResults(question, cycles, evaluator_model_list)
                self.question_result_list.append(question_result)
        self.score = None

    def get_cycle(self, question_id, cycle_number):
        result = None
        for question_result in self.question_result_list:
            if question_result.question.id == question_id:
                result = question_result.get_cycle(cycle_number)
                break
        return result

    def calculate_scores(self, status_report):
        cumulative_score = 0.0
        score_count = 0
        for question_result in self.question_result_list:
            question_score = question_result.calculate_scores(status_report)
            if question_score is not None:
                cumulative_score += question_score
                score_count += 1
        if score_count > 0:
            self.score = cumulative_score / score_count
        return self.score

    def get_cycle_scores(self):
        result = []
        cycle_name_list = self.question_result_list[0].get_cycle_names()
        for cycle_name in cycle_name_list:
            accumulated_score = ScoreAccumulator()
            for question_result in self.question_result_list:
                question_result.add_score_for_cycle(cycle_name, accumulated_score)
            cycle_score = CycleScore(cycle_name, accumulated_score.get_score())
            result.append(cycle_score)
        return result

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.question_result_list is not None:
            index = 0
            for question_result in self.question_result_list:
                result["question_result_list"][index] = question_result.to_dict()
                index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        question_result_list = dictionary.get("question_result_list", None)
        if question_result_list is not None:
            dictionary.pop("question_result_list", None)
            question_result_list = [QuestionResults.from_dict(result) for result in question_result_list]
            dictionary["question_result_list"] = question_result_list
        result = LocationResults(**dictionary)
        return result


class ModelResults:
    def __init__(self, directory, model_name, location_token_index_list, question_list, cycles, evaluator_model_list,
                 location_list=None, test_exception_list=None, evaluation_exception_list=None):
        self.directory = directory
        self.model_name = model_name
        self.location_list = location_list
        self.test_exception_list = test_exception_list
        if self.test_exception_list is None:
            self.test_exception_list = []
        self.failed_test_count = 0
        self.evaluation_exception_list = evaluation_exception_list
        if self.evaluation_exception_list is None:
            self.evaluation_exception_list = []
        self.failed_evaluation_count = 0
        if self.location_list is None:
            self.location_list = []
            for location in location_token_index_list:
                location_result = LocationResults(location, question_list, cycles, evaluator_model_list)
                self.location_list.append(location_result)
        os.makedirs(directory, exist_ok=True)

    def get_cycle(self, location_name, question_id, cycle_number):
        result = None
        for location in self.location_list:
            if location.location_token_position == location_name:
                result = location.get_cycle(question_id, cycle_number)
                break
        return result

    def calculate_scores(self):
        status_report = StatusReport(self.model_name, self.failed_test_count, self.failed_evaluation_count)
        for location in self.location_list:
            location.calculate_scores(status_report)
        status_report.print()
        return status_report

    def get_location_scores(self):
        result = []
        for location in self.location_list:
            cycle_scores = location.get_cycle_scores()
            location_score = LocationScore(location.location_token_position, location.score, cycle_scores)
            result.append(location_score)
        return result

    def add_test_exception(self, location_name, question_id, cycle_number, attempt, exception):
        print("Test Exception: ", str(exception))
        exception_report = TestResultExceptionReport(location_name, question_id, cycle_number, attempt,str(exception))
        self.test_exception_list.append(exception_report)
        if attempt == PROMPT_RETRIES - 1:
            self.failed_test_count += 1
            print("Failed Test Count: ", self.failed_test_count)

    def add_evaluation_exception(self, location_name, question_id, cycle_number, attempt, evaluation_model_name,
                                 exception):
        print("Evaluation Exception: ", str(exception))
        exception_report = EvaluationExceptionReport(location_name, question_id, cycle_number, attempt,
                                                     evaluation_model_name, str(exception))
        self.evaluation_exception_list.append(exception_report)
        if attempt == PROMPT_RETRIES - 1:
            self.failed_evaluation_count += 1
            print("Failed Evaluation Count: ", self.failed_evaluation_count)

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        if self.location_list is not None:
            index = 0
            for location in self.location_list:
                result["location_list"][index] = location.to_dict()
                index += 1
        if self.test_exception_list is not None:
            index = 0
            for exception_report in self.test_exception_list:
                result["test_exception_list"][index] = exception_report.to_dict()
                index += 1
        if self.evaluation_exception_list is not None:
            index = 0
            for exception_report in self.evaluation_exception_list:
                result["evaluation_exception_list"][index] = exception_report.to_dict()
                index += 1
        result.pop("directory", None)
        return result

    @staticmethod
    def from_dict(dictionary):
        location_list = dictionary.get("location_list", None)
        if location_list is not None:
            dictionary.pop("location_list", None)
            location_list = [LocationResults.from_dict(location) for location in location_list]
            dictionary["location_list"] = location_list
        test_exception_list = dictionary.get("test_exception_list", None)
        if test_exception_list is not None:
            dictionary.pop("test_exception_list", None)
            test_exception_list = [TestResultExceptionReport.from_dict(report) for report in test_exception_list]
            dictionary["test_exception_list"] = test_exception_list
        evaluation_exception_list = dictionary.get("evaluation_exception_list", None)
        if evaluation_exception_list is not None:
            dictionary.pop("evaluation_exception_list", None)
            evaluation_exception_list = [EvaluationExceptionReport.from_dict(report) for report in evaluation_exception_list]
            dictionary["evaluation_exception_list"] = evaluation_exception_list
        result = ModelResults(**dictionary)
        return result


class BaseTestResultAction:
    def __init__(self, results, model_name, location_name, question_id, cycle_number):
        self.results = results
        self.model_name = model_name
        self.location_name = location_name
        self.question_id = question_id
        self.cycle_number = cycle_number

    def get_model_results(self):
        result = self.results.get_model_results(self.model_name)
        return result

    def get_cycle(self):
        result = self.get_model_results().get_cycle(self.location_name, self.question_id, self.cycle_number)
        return result

    def execute(self):
        raise NotImplementedError


class SetTestResultAction(BaseTestResultAction):
    def __init__(self, results, model_name, location_name, question_id, cycle_number, generated_answer):
        super().__init__(results, model_name, location_name, question_id, cycle_number)
        self.generated_answer = generated_answer

    def execute(self):
        self.get_cycle().set_generated_answer(self.generated_answer)


class SetEvaluatorResultAction(BaseTestResultAction):
    def __init__(self, results, model_name, location_name, question_id, cycle_number, evaluator_model_name, passed):
        super().__init__(results, model_name, location_name, question_id, cycle_number)
        self.evaluator_model_name = evaluator_model_name
        self.passed = passed

    def execute(self):
        self.get_cycle().set_evaluator_result(self.evaluator_model_name, self.passed)


class AddTestExceptionAction(BaseTestResultAction):
    def __init__(self, results, model_name, location_name, question_id, cycle_number, attempt, exception):
        super().__init__(results, model_name, location_name, question_id, cycle_number)
        self.attempt = attempt
        self.exception = exception

    def execute(self):
        self.get_model_results().add_test_exception(self.location_name, self.question_id, self.cycle_number, self.attempt, self.exception)


class AddEvaluationExceptionAction(BaseTestResultAction):
    def __init__(self, results, model_name, location_name, question_id, cycle_number, evaluation_model_name, attempt, exception):
        super().__init__(results, model_name, location_name, question_id, cycle_number)
        self.evaluation_model_name = evaluation_model_name
        self.attempt = attempt
        self.exception = exception

    def execute(self):
        self.get_model_results().add_evaluation_exception(self.location_name, self.question_id, self.cycle_number,
                                                          self.evaluation_model_name, self.attempt, self.exception)


class TestResults:
    def __init__(self, test_result_directory, model_results_list=None):
        self.test_result_directory = test_result_directory
        self.model_results_list = model_results_list
        if self.model_results_list is None:
            self.model_results_list = []
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
            action.execute()

    def start(self):
        self.started = True
        self.timer = threading.Timer(interval=STATUS_REPORT_INTERVAL, function=self.update_and_report_status)
        self.timer.start()

    def get_model_results(self, model_name):
        result = None
        for model_results in self.model_results_list:
            if model_results.model_name == model_name:
                result = model_results
                break
        return result

    def get_model_results_directory(self, model_name):
        model_result = self.get_model_results(model_name)
        result = model_result.directory
        return result

    def copy_model_results_list(self):
        result = None
        result = copy.deepcopy(self.model_results_list)
        return result

    def add_model(self, model_name, location_list, question_list, cycles, evaluator_model_list):
        directory = os.path.join(self.test_result_directory, model_name)
        model_results = ModelResults(directory, model_name, location_list, question_list, cycles, evaluator_model_list)
        self.model_results_list.append(model_results)
        return model_results

    def set_test_result(self, model_name, location_name, question_id, cycle_number, generated_answer):
        action = SetTestResultAction(self, model_name, location_name, question_id, cycle_number, generated_answer)
        self.add_action(action)

    def set_evaluator_result(self, model_name, location_name, question_id, cycle_number, evaluator_model_name, passed):
        action = SetEvaluatorResultAction(self, model_name, location_name, question_id, cycle_number,
                                          evaluator_model_name, passed)
        self.add_action(action)

    def add_test_exception(self, model_name, location_name, question_id, cycle_number, attempt, exception):
        action = AddTestExceptionAction(self, model_name, location_name, question_id, cycle_number, attempt, exception)
        self.add_action(action)

    def add_evaluation_exception(self, model_name, location_name, question_id, cycle_number, evaluation_model_name,
                                 attempt, exception):
        action = AddEvaluationExceptionAction(self, model_name, location_name, question_id, cycle_number,
                                              evaluation_model_name, attempt, exception)
        self.add_action(action)

    def calculate_scores(self):
        for model_result in self.model_results_list:
            model_result.calculate_scores()

    def update_and_report_status(self):
        if not self.started:
            return
        self.execute_actions()
        current_results_list = self.model_results_list
        model_status_list = []
        finished = True
        for model_results in current_results_list:
            status_report = model_results.calculate_scores()
            if not status_report.is_finished():
                finished = False
            model_status_list.append(status_report)
        if not finished:
            self.write_full_results(current_results_list)
            self.timer = threading.Timer(interval=STATUS_REPORT_INTERVAL, function=self.update_and_report_status)
            self.timer.start()
        else:
            model_score_list = []
            for model_results in current_results_list:
                location_scores = model_results.get_location_scores()
                model_score = ModelScore(model_results.model_name, location_scores)
                plot_name = model_results.model_name + "_plot.png"
                plot_file_name = os.path.join(self.test_result_directory, plot_name)
                model_score.write_plot(plot_file_name)
                model_score_list.append(model_score)
            test_model_scores = TestModelScores(model_score_list)
            test_model_scores.write_to_file(os.path.join(self.test_result_directory, "model_scores.json"))
            self.write_full_results(current_results_list)
            print("All tests are finished")

    def write_full_results(self, results_list):
        results_list = copy.deepcopy(results_list)
        for model_results in results_list:
            full_results_name = model_results.model_name + "_full_results.json"
            file_name = os.path.join(self.test_result_directory, full_results_name)
            with open(file_name, "w") as file:
                results_dict = model_results.to_dict()
                json.dump(results_dict, file, indent=4)


if __name__ == '__main__':
    model_scores_path = os.environ.get("MODEL_SCORES")
    with open(model_scores_path, "r") as file:
        model_scores_dict = json.load(file)
        test_model_scores = TestModelScores.from_dict(model_scores_dict)
        for model_score in test_model_scores.model_scores:
            print("Model: ", model_score.model_name)
            plot_name = model_score.model_name + "_plot.png"
            model_score.write_plot(plot_name)
        print("done")
        exit(0)
