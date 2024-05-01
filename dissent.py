import os

from test_results import ModelResults


class EvaluatorReport:
    def __init__(self, model_name, total_score_count, test_count=0, agreed_test_count=0, disagreed_test_count=0):
        self.model_name = model_name
        self.total_score_count = total_score_count
        self.test_count = test_count
        self.agreed_test_count = agreed_test_count
        self.disagreed_test_count = disagreed_test_count

    def add_test(self, trial_answer, evaluator_answer):
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


class DissentReport:
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
        self.dissenting_evaluator_report = {}

    def process(self):
        print("Processing results")
        for trial in self.trials_with_dissent_list:
            for evaluator in trial.evaluator_results:
                evaluator_report = self.dissenting_evaluator_report.get(evaluator.model_name, None)
                if evaluator_report is None:
                    evaluator_report = EvaluatorReport(evaluator.model_name, len(self.trial_list))
                    self.dissenting_evaluator_report[evaluator.model_name] = evaluator_report
                evaluator_report.add_test(trial.passed, evaluator.passed)
        self.print_dissenting_evaluator_report()

    def print_dissenting_evaluator_report(self):
        print("Dissenting Evaluator Report")
        for evaluator_report in self.dissenting_evaluator_report.values():
            score = evaluator_report.get_percent_wrong()
            print(evaluator_report.model_name + " % wrong: " + str(score) + "%")


if __name__ == '__main__':
    full_results_path = os.environ.get("FULL_RESULTS_PATH", "gpt-3.5-turbo-0125_full_results.json")
    report = DissentReport(full_results_path)
    report.process()
    print("done")
    exit(0)

