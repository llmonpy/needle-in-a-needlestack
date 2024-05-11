import copy
import json
import os

from test_results import ModelResults, ModelScore, TestModelScores, MODEL_SCORES_FILE


class ReplaceQuestionInTrial:
    def __init__(self, test_result_directory, source_file_path, original_file_path_list, source_question_id, original_question_id):
        self.test_result_directory = test_result_directory
        self.source_file_path = source_file_path
        self.original_file_path_list = original_file_path_list
        self.source_question_id = source_question_id
        self.original_question_id = original_question_id

    def process(self):
        source_results = ModelResults.from_file(self.source_file_path)
        model_score_list = []
        new_results_list = []
        for original_file_path in self.original_file_path_list:
            new_results = ModelResults.from_file(original_file_path)
            new_results.replace_question(source_results, self.source_question_id, self.original_question_id)
            new_results.calculate_scores()
            location_scores = new_results.get_location_scores()
            model_score = ModelScore(new_results.model_name, new_results.date_string,
                                     new_results.repeat_question_limerick_count,
                                     new_results.limerick_count_in_prompt, location_scores,
                                     new_results.number_of_trials_per_location)
            plot_name = new_results.model_name + "_trial_plot.png"
            plot_file_name = os.path.join(self.test_result_directory, plot_name)
            model_score.write_trial_plot(plot_file_name)
            plot_name = new_results.model_name + "_question_plot_"
            plot_file_name = os.path.join(self.test_result_directory, plot_name)
            model_score.write_question_plot(plot_file_name)
            model_score_list.append(model_score)
            new_results_list.append(new_results)
        test_model_scores = TestModelScores(model_score_list)
        test_model_scores.write_to_file(os.path.join(self.test_result_directory, MODEL_SCORES_FILE))
        self.write_full_results(new_results_list)

    def write_full_results(self, results_list):
        results_list = copy.deepcopy(results_list)
        for model_results in results_list:
            full_results_name = model_results.model_name + "_full_results.json"
            file_name = os.path.join(self.test_result_directory, full_results_name)
            with open(file_name, "w") as file:
                results_dict = model_results.to_dict()
                json.dump(results_dict, file, indent=4)


if __name__ == '__main__':
    ReplaceQuestionInTrial("fix/sonnet","fix/sonnet/sonnet.json", ["fix/sonnet/original_sonnet.json"],
                           84044,99730).process()
    os._exit(0)


