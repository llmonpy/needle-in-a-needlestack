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
import os
import sys

from test_config import get_latest_test_directory
from test_results import ModelResults, REEVALUATION_FILE_PREFIX


class EvaluatorReport:
    def __init__(self, model_name, evaluation_count=0, agreed_count=0, disagreed_count=0):
        self.model_name = model_name
        self.evaluation_count = evaluation_count
        self.agreed_count = agreed_count
        self.disagreed_count = disagreed_count

    def add_evaluation(self, trial_answer, evaluator_answer):
        self.evaluation_count += 1
        if trial_answer == evaluator_answer:
            self.agreed_count += 1
        else:
            self.disagreed_count += 1

    def get_percent_wrong(self):
        result = round((self.disagreed_count / self.evaluation_count) * 100)
        return result


class ModelDissentReport:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_results = ModelResults.from_file(self.file_path)
        self.trial_list = self.model_results.get_all_trial_results()

    def get_model_name(self):
        return self.model_results.model_name

    def get_trial_count(self):
        return len(self.trial_list)

    def process(self, dissent_report):
        print("Processing results for " + self.file_path)
        for trial in self.trial_list:
            for evaluator in trial.evaluator_results:
                dissent_report.add_evaluation(evaluator.model_name, trial.passed, evaluator.passed)


class DissentReport:
    def __init__(self, directory, model_full_results_file_list):
        self.directory = directory
        self.model_dissent_reports = []
        self.evaluator_grades = {}
        for file_name in model_full_results_file_list:
            file_path = os.path.join(directory, file_name)
            model_dissent_report = ModelDissentReport(file_path)
            self.model_dissent_reports.append(model_dissent_report)

    def add_evaluation(self, model_name, trial_passed, evaluator_passed):
        evaluator_report = self.evaluator_grades.get(model_name, None)
        if evaluator_report is None:
            evaluator_report = EvaluatorReport(model_name)
            self.evaluator_grades[model_name] = evaluator_report
        evaluator_report.add_evaluation(trial_passed, evaluator_passed)

    def process(self):
        for model_dissent_report in self.model_dissent_reports:
            model_dissent_report.process(self)
        self.print_evaluator_grade_report()

    def print_evaluator_grade_report(self):
        output_file_path = os.path.join(self.directory, "evaluator_grades.txt")
        with open(output_file_path, "w") as file:
            print("Dissenting Evaluator Report\n")
            for evaluator_report in self.evaluator_grades.values():
                score = evaluator_report.get_percent_wrong()
                message = evaluator_report.model_name + " % wrong: " + str(score) + "%\n"
                file.write(message)
                print(message)
        print("done")

    @staticmethod
    def create_from_original_results(directory):
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if not file.startswith(REEVALUATION_FILE_PREFIX) and
                                        file.endswith("full_results.json")]
        result = DissentReport(directory, model_full_results_file_list)
        return result

    @staticmethod
    def create_from_revaluator_results(directory, prefix):
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if file.startswith(prefix)
                                        and file.endswith("full_results.json")]
        result = DissentReport(directory, model_full_results_file_list)
        return result

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        full_results_path = sys.argv[1]
    else:
        full_results_path = get_latest_test_directory()
    report = DissentReport.create_from_original_results(full_results_path)
    report.process()
    exit(0)


