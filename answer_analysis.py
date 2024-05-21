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

import copy
import json
import os
import sys
import nothingpy

from limerick import Limerick, FULL_QUESTION_FILE
from test_config import get_latest_test_directory
from test_results import ModelResults, QuestionAnswerCollector


class QuestionAnswerList:
    def __init__(self, question, passed_answer_list=None, failed_answer_list=None):
        self.question = question
        self.passed_answer_list = passed_answer_list if passed_answer_list is not None else []
        self.failed_answer_list = failed_answer_list if failed_answer_list is not None else []
        self.passed_answer_dict = {}
        self.failed_answer_dict = {}

    def add_answer(self, answer, passed):
        if passed:
            self.passed_answer_dict[answer] = answer
        else:
            self.failed_answer_dict[answer] = answer

    def finish(self):
        self.passed_answer_list = list(self.passed_answer_dict.values())
        self.failed_answer_list = list(self.failed_answer_dict.values())

    def to_dict(self):
        result = copy.copy(vars(self))
        result.pop("passed_answer_dict", None)
        result.pop("failed_answer_dict", None)
        result["question"] = self.question.to_dict()
        return result

    @staticmethod
    def from_dict(dictionary):
        question = Limerick.from_dict(dictionary["question"])
        passed_answer_list = dictionary.get("passed_answer_list", None)
        failed_answer_list = dictionary.get("failed_answer_list", None)
        result = QuestionAnswerList(question, passed_answer_list, failed_answer_list)
        return result


class AnswerAnalysis(QuestionAnswerCollector):
    def __init__(self, question_answer_list=None):
        self.question_answer_list = question_answer_list if question_answer_list is not None else []

    def add_question(self, question):
        question_answer = QuestionAnswerList(question)
        self.question_answer_list.append(question_answer)

    def add_answer(self, question_id, answer, passed):
        for question_answer in self.question_answer_list:
            if question_answer.question.id == question_id:
                question_answer.add_answer(answer, passed)
                break

    def finish(self):
        for question_answer in self.question_answer_list:
            question_answer.finish()

    def write_to_file(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self):
        result = copy.copy(vars(self))
        index = 0
        for question_answer in self.question_answer_list:
            result["question_answer_list"][index] = question_answer.to_dict()
            index += 1
        return result

    @staticmethod
    def from_dict(dictionary):
        question_answer_list = dictionary.get("question_answer_list", None)
        if question_answer_list is not None:
            dictionary.pop("question_answer_list", None)
            question_answer_list = [QuestionAnswerList.from_dict(question_answer_dict) for question_answer_dict in
                                    question_answer_list]
            dictionary["question_answer_list"] = question_answer_list
        result = AnswerAnalysis(**dictionary)
        return result

    @staticmethod
    def create_from_test_runs(directory):
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if file.endswith("full_results.json")]
        result = AnswerAnalysis.create(directory, model_full_results_file_list)
        return result

    @staticmethod
    def create_from_revaluator_results(directory, prefix):
        all_files = os.listdir(directory)
        model_full_results_file_list = [file for file in all_files if file.startswith(prefix)
                                        and file.endswith("full_results.json")]
        result = AnswerAnalysis.create(directory, model_full_results_file_list)
        return result

    @staticmethod
    def create(full_results_path, model_full_results_file_list):
        result = AnswerAnalysis()
        with open(FULL_QUESTION_FILE, "r") as file:
            question_dict_list = json.load(file)
        question_list = [Limerick.from_dict(question_dict) for question_dict in question_dict_list]
        for question in question_list:
            result.add_question(question)
        for file_name in model_full_results_file_list:
            results_file_path = os.path.join(full_results_path, file_name)
            model_results = ModelResults.from_file(results_file_path)
            model_results.collect_question_answers(result)
        return result


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        full_results_path = sys.argv[1]
    else:
        full_results_path = get_latest_test_directory()
    analyzer = AnswerAnalysis.create_from_test_runs(full_results_path)
    analyzer.finish()
    results_path = os.path.join(full_results_path, "answer_analysis.json")
    analyzer.write_to_file(results_path)
    print("Finished writing results to " + results_path)
    os._exit(0)



