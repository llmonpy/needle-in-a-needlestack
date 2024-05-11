from test_results import ModelResults


class ReplaceQuestionInTrial:
    def __init__(self, source_file_path, original_file_path_list, replace_question_id):
        self.source_file_path = source_file_path
        self.original_file_path_list = original_file_path_list
        self.replace_question_id = replace_question_id

    def process(self):
        source_results = ModelResults.from_file(self.source_file_path)
        model_score_list = []
        for original_file_path in self.original_file_path_list:
            original_results = ModelResults.from_file(original_file_path)
            original_results.replace_question(source_results, self.replace_question_id)
