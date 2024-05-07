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
import sys

from answer_analysis import AnswerAnalysis
from dissent import DissentReport
from evaluator import DefaultEvaluator
from limerick import FULL_QUESTION_FILE, Limerick
from test_config import DEFAULT_TEST_CONFIG
from test_results import ModelResults, ORIGINAL_MODEL_NAME, REPLACEMENT_MODEL_NAME, REEVALUATION_FILE_PREFIX
from test_status import TestStatus


class AnswerReevaluator:
    def __init__(self, directory, evaluator_model_list, replace_evaluator_model=None):
        self.directory = directory
        self.evaluator_model_list = evaluator_model_list
        self.model_results_list = []
        self.changed_evaluation_list = []
        with open(FULL_QUESTION_FILE, "r") as file:
            question_dict_list = json.load(file)
        self.current_question_list = [Limerick.from_dict(question_dict) for question_dict in question_dict_list]
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if not file.startswith(REEVALUATION_FILE_PREFIX) and
                                        file.endswith("full_results.json")]
        model_name_list = []
        for file_name in model_full_results_file_list:
            results_file_path = os.path.join(full_results_path, file_name)
            model_results = ModelResults.from_file(results_file_path)
            model_results.update_questions(self.current_question_list)
            model_results.update_evaluator_models(replace_evaluator_model)
            model_name_list.append(model_results.model_name)
            self.model_results_list.append(model_results)
        self.test_status = TestStatus(None, evaluator_model_list, model_name_list)

    def reevaluate_generated_answers(self):
        futures_list = []
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1000)
        for model_results in self.model_results_list:
            futures_list += model_results.reevaluate_generated_answers(thread_pool, self.test_status,
                                                                       DefaultEvaluator(self.test_status,
                                                                                        self.evaluator_model_list))
        self.test_status.start(self)
        for future in concurrent.futures.as_completed(futures_list):
            trial, changed_result = future.result()
            if changed_result:
                trial = copy.deepcopy(trial)
                self.changed_evaluation_list.append(trial)
        return futures_list

    def all_tests_finished(self):
        for model_results in self.model_results_list:
            full_results_name = REEVALUATION_FILE_PREFIX + model_results.model_name + "_full_results.json"
            file_name = os.path.join(self.directory, full_results_name)
            with open(file_name, "w") as file:
                results_dict = model_results.to_dict()
                json.dump(results_dict, file, indent=4)
        if len(self.changed_evaluation_list) > 0:
            print("Some evaluations changed")
            file_name = os.path.join(self.directory, "changed_evaluations.json")
            with open(file_name, "w") as file:
                trial_dict_list = []
                for trial in self.changed_evaluation_list:
                    trial_dict = trial.to_dict()
                    trial_dict_list.append(trial_dict)
                json.dump(trial_dict_list, file, indent=4)
        print("All tests are finished")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        full_results_path = sys.argv[1]
    else:
        full_results_path = "answer_examples"
    test_config = DEFAULT_TEST_CONFIG
    reevaluator = AnswerReevaluator(full_results_path, test_config.evaluator_model_list,
                                    {ORIGINAL_MODEL_NAME:"gpt-3.5-turbo-0125",
                                     REPLACEMENT_MODEL_NAME:"open-mixtral-8x22b"})
    reevaluator.reevaluate_generated_answers()
    analyzer = AnswerAnalysis.create_from_revaluator_results(full_results_path, REEVALUATION_FILE_PREFIX)
    analyzer.finish()
    results_path = os.path.join(full_results_path, "reeval_answer_analysis.json")
    analyzer.write_to_file(results_path)
    dissent_report = DissentReport.create_from_revaluator_results(full_results_path, REEVALUATION_FILE_PREFIX)
    dissent_report.process()
    print("Finished")
    exit(0)
