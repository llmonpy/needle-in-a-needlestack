import os
import sys

from test_config import get_latest_test_directory
from test_results import ModelResults, REEVALUATION_FILE_PREFIX


class ModelQuestionTrialVarianceReport:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_results = ModelResults.from_file(self.file_path)
        self.question_trial_list = self.model_results.get_all_question_results()
        self.number_with_dissent = 0
        for question in self.question_trial_list:
            if question.trials_have_dissent():
                self.number_with_dissent += 1

    def get_percent_with_dissent(self):
        result = round((self.number_with_dissent / len(self.question_trial_list)) * 100)
        return result


class QuestionTrialVarianceReport:
    def __init__(self, directory, model_full_results_file_list):
        self.directory = directory
        self.evaluator_grades = {}
        output_file_path = os.path.join(self.directory, "question_variance.txt")
        with open(output_file_path, "w") as file:
            for file_name in model_full_results_file_list:
                file_path = os.path.join(directory, file_name)
                model_dissent_report = ModelQuestionTrialVarianceReport(file_path)
                message = "Model: " + model_dissent_report.model_results.model_name + " Percent with variance: " \
                      + str(model_dissent_report.get_percent_with_dissent()) + "%"
                file.write(message)
                print(message)
        print("done")


    @staticmethod
    def create_from_original_results(directory):
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if not file.startswith(REEVALUATION_FILE_PREFIX) and
                                        file.endswith("full_results.json")]
        result = QuestionTrialVarianceReport(directory, model_full_results_file_list)
        return result


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        full_results_path = sys.argv[1]
    else:
        full_results_path = get_latest_test_directory()
    report = QuestionTrialVarianceReport.create_from_original_results(full_results_path)
    os._exit(0)
