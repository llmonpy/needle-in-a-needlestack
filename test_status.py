
class EvaluatorStatus:
    def __init__(self, evaluator_model_name):
        self.evaluator_model_name = evaluator_model_name
        self.evaluation_count = 0
        self.evaluation_finished_count = 0
        self.evaluation_exception_count = 0
        self.evaluation_failure_count = 0

    def add_evaluation(self):
        self.evaluation_count += 1

    def evaluation_finished(self):
        self.evaluation_finished_count += 1

    def add_evaluation_exception(self):
        self.evaluation_exception_count += 1

    def add_evaluation_failure(self):
        self.evaluation_failure_count += 1


class ModelTestStatus:
    def __init__(self, model_name):
        self.model_name = model_name
        self.test_count = 0
        self.answers_generated_count = 0
        self.test_finished_count = 0
        self.test_exception_count = 0
        self.answer_generation_failure_count = 0
        self.evaluation_failure_count = 0

    def add_test(self):
        self.test_count += 1

    def test_answer_generated(self):
        self.answers_generated_count += 1

    def test_finished(self):
        self.test_finished_count += 1

    def add_test_exception(self):
        self.test_exception_count += 1

    def add_answer_generation_failure(self):
        self.answer_generation_failure_count += 1

    def add_evaluation_failure(self):
        self.evaluation_failure_count += 1


class TestStatus:
    def __init__(self, evaluator_model_name_list):
        self.evaluator_model_name_list = evaluator_model_name_list
        self.test_count = 0
        self.answers_generated = 0
        self.test_finished = 0
        self.evaluations_required = 0
        self.evaluations_completed = 0
        self.model_test_status_list = []
        self.model_evaluation_status_list = []
        self.test_exception_list = []
        self.evaluation_exception_list = []
        self.test_failures = 0
        self.evaluation_failures = 0

    def add_test(self, model_name):
        self.test_count += 1

    def test_answer_generated(self, model_name):
        self.answers_generated_count += 1

    def test_finished(self, model_name):
        self.test_finished_count += 1

    def add_test_exception(self, model_name, exception, give_up=False):
        self.test_exception_count += 1

    def add_evaluation_exception(self, model_name, evaluator_model_name, exception, give_up=False):
        self.evaluation_exception_count += 1

    def add_answer_generation_failure(self, model_name):
        self.answer_generation_failure_count += 1

    def add_evaluation_failure(self, evaluator_model_name):
        self.evaluation_failure_count += 1
