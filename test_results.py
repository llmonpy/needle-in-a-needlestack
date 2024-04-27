import copy
import json
import os
import threading
import matplotlib.pyplot as plot

from limerick import Limerick

STATUS_REPORT_INTERVAL = 5 # seconds


class StatusReport:
    def __init__(self, model_name):
        self.model_name = model_name
        self.test_count = 0
        self.answered_test_count = 0
        self.finished_test_count = 0
        self.evaluator_count = 0
        self.finished_evaluator_count = 0

    def add_test(self, has_answer, is_finished):
        self.test_count += 1
        if has_answer:
            self.answered_test_count += 1
        if is_finished:
            self.finished_test_count += 1

    def add_evaluator_test(self, finished):
        self.evaluator_count += 1
        if finished:
            self.finished_evaluator_count += 1

    def is_finished(self):
        result = self.test_count == self.finished_test_count
        return result

    def print(self):
        print("Model: ", self.model_name)
        print("Tests: ", self.test_count, " Answered: ", self.answered_test_count, " Finished: ", self.finished_test_count)
        print("Evaluators: ", self.evaluator_count, " Finished: ", self.finished_evaluator_count)
        print("--------------------")


class LocationScore:
    def __init__(self, location_token_position, score):
        self.location_token_position = location_token_position
        self.score = score

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
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

    def write_plot(self, plot_file_name):
        labels = [location.location_token_position for location in self.location_scores]
        values = [round(location.score * 100) for location in self.location_scores]

        plot.figure(figsize=(10, 10))
        plot.plot(labels, values, marker='o')
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
            location_scores = [LocationScore.from_dict(result) for result in location_scores]
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
            status_report.add_evaluator_test(evaluator_result.is_finished())
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
                 location_list=None):
        self.directory = directory
        self.model_name = model_name
        self.location_list = location_list
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
        status_report = StatusReport(self.model_name)
        for location in self.location_list:
            location.calculate_scores(status_report)
        status_report.print()
        return status_report

    def get_location_scores(self):
        result = []
        for location in self.location_list:
            location_score = LocationScore(location.location_token_position, location.score)
            result.append(location_score)
        return result

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.location_list is not None:
            index = 0
            for location in self.location_list:
                result["location_list"][index] = location.to_dict()
                index += 1
        result.pop("directory", None)
        return result

    @staticmethod
    def from_dict(dictionary):
        location_list = dictionary.get("location_list", None)
        if location_list is not None:
            dictionary.pop("location_list", None)
            location_list = [LocationResults.from_dict(result) for result in location_list]
            dictionary["location_list"] = location_list
        result = ModelResults(**dictionary)
        return result


class TestResults:
    def __init__(self, test_result_directory, model_results_list=None):
        self.test_result_directory = test_result_directory
        self.model_results_list = model_results_list
        if self.model_results_list is None:
            self.model_results_list = []
        self.results_list_lock = threading.Lock()
        self.started = False
        self.timer = None

    def start(self):
        self.started = True
        self.timer = threading.Timer(interval=STATUS_REPORT_INTERVAL, function=self.report_status)
        self.timer.start()

    def get_model_results(self, model_name):
        result = None
        with self.results_list_lock:
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
        with self.results_list_lock:
            result = copy.deepcopy(self.model_results_list)
        return result

    def add_model(self, model_name, location_list, question_list, cycles, evaluator_model_list):
        directory = os.path.join(self.test_result_directory, model_name)
        model_results = ModelResults(directory, model_name, location_list, question_list, cycles, evaluator_model_list)
        self.model_results_list.append(model_results)
        return model_results

    def set_test_result(self, model_name, location_name, question_id, cycle_number, generated_answer):
        # this is thread safe because only one thread will set each test_result.  Need to use a lock to get model
        # results because the status reporter copies the current results before processing for thread safety
        model_result = self.get_model_results(model_name)
        cycle = model_result.get_cycle(location_name, question_id, cycle_number)
        cycle.set_generated_answer(generated_answer)

    def set_evaluator_result(self, model_name, location_name, question_id, cycle_number, evaluator_model_name, passed):
        # this is thread safe because only one thread will set each evaluator_result.  Need to use a lock to get model
        # results because the status reporter copies the current results before processing for thread safety
        model_result = self.get_model_results(model_name)
        cycle = model_result.get_cycle(location_name, question_id, cycle_number)
        cycle.set_evaluator_result(evaluator_model_name, passed)

    def calculate_scores(self):
        # this is not thread safe, but is only called after all tests are finished
        for model_result in self.model_results_list:
            model_result.calculate_scores()

    def report_status(self):
        if not self.started:
            return
        current_results_list = self.copy_model_results_list()
        model_status_list = []
        finished = True
        for model_results in current_results_list:
            status_report = model_results.calculate_scores()
            if not status_report.is_finished():
                finished = False
            model_status_list.append(status_report)
        if not finished:
            self.timer = threading.Timer(interval=STATUS_REPORT_INTERVAL, function=self.report_status)
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
        for model_results in results_list:
            full_results_name = model_results.model_name + "_full_results.json"
            file_name = os.path.join(self.test_result_directory, full_results_name)
            with open(file_name, "w") as file:
                results_dict = model_results.to_dict()
                json.dump(results_dict, file, indent=4)

