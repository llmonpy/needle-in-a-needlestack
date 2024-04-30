

class BaseStatusReport:
    def __init__(self, failed_test_count=0, failed_evaluation_count=0):
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
        raise NotImplementedError


class BaseTestResults:
    def __init__(self):
        pass

    def set_test_result(self, model_name, location_name, question_id, cycle_number, generated_answer):
        raise NotImplementedError

    def set_evaluator_result(self, model_name, location_name, question_id, cycle_number, evaluator_model_name, passed):
        raise NotImplementedError

    def add_test_exception(self, model_name, location_name, question_id, cycle_number, attempt, exception):
        raise NotImplementedError

    def add_evaluation_exception(self, model_name, location_name, question_id, cycle_number, evaluation_model_name,
                                 attempt, exception):
        raise NotImplementedError
