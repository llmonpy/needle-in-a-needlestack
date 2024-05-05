import os

from test_results import ModelResults


class EvaluatorReport:
    def __init__(self, model_name, total_score_count, test_count=0, agreed_test_count=0, disagreed_test_count=0):
        self.model_name = model_name
        self.total_score_count = total_score_count
        self.test_count = test_count
        self.agreed_test_count = agreed_test_count
        self.disagreed_test_count = disagreed_test_count

    def add_trial(self, trial_answer, evaluator_answer):
        self.test_count += 1
        if trial_answer == evaluator_answer:
            self.agreed_test_count += 1
        else:
            self.disagreed_test_count += 1

    def get_percent_wrong_when_there_is_dissent(self):
        result = round((self.disagreed_test_count / self.test_count) * 100)
        return result

    def get_percent_wrong(self):
        result = round((self.disagreed_test_count / self.total_score_count) * 100)
        return result


class ModelDissentReport:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model_results = ModelResults.from_file(self.file_path)
        self.trial_list = self.model_results.get_all_trial_results()
        self.trials_with_dissent_list = []
        self.trials_with_concerning_dissent_list = []
        for trial in self.trial_list:
            if trial.has_dissent():
                self.trials_with_dissent_list.append(trial)
            if trial.has_concerning_dissent():
                self.trials_with_concerning_dissent_list.append(trial)

    def get_model_name(self):
        return self.model_results.model_name

    def get_trial_count(self):
        return len(self.trial_list)

    def process(self, dissent_report):
        print("Processing results for " + self.file_path)
        for trial in self.trials_with_dissent_list:
            for evaluator in trial.evaluator_results:
                dissent_report.add_trial(evaluator.model_name, self.get_trial_count(), trial.passed, evaluator.passed)


class DissentReport:
    def __init__(self, directory):
        self.directory = directory
        self.model_dissent_reports = []
        all_files = os.listdir(full_results_path)
        model_full_results_file_list = [file for file in all_files if file.endswith("full_results.json")]
        self.evaluator_grades = {}
        for file_name in model_full_results_file_list:
            file_path = os.path.join(full_results_path, file_name)
            model_dissent_report = ModelDissentReport(file_path)
            self.model_dissent_reports.append(model_dissent_report)

    def add_trial(self, model_name, trial_count, trial_passed, evaluator_passed):
        evaluator_report = self.evaluator_grades.get(model_name, None)
        if evaluator_report is None:
            evaluator_report = EvaluatorReport(model_name, trial_count)
            self.evaluator_grades[model_name] = evaluator_report
        evaluator_report.add_trial(trial_passed, evaluator_passed)

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

if __name__ == '__main__':
    full_results_path = os.environ.get("FULL_RESULTS_PATH")
    report = DissentReport(full_results_path)
    report.process()
    print("done")
    exit(0)

