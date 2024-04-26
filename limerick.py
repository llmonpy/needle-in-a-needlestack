import copy
import json
import random
import tiktoken

NUMBER_OF_TEST_QUESTIONS_TO_GENERATE = 3
GENERATED_QUESTION_FILE = "questions.json"
FULL_QUESTION_FILE = "full_questions.json"
LIMERICK_DATASET_FILE = "limerick_dataset_oedilf_v3.json"

class Limerick:
    def __init__(self, id, author, text, question=None, answer=None, tokens=None, token_count=None, target_location=0):
        self.id = id
        self.author = author
        self.text = text
        self.question = question
        self.answer = answer
        self.tokens = tokens
        self.token_count = token_count
        self.target_location = target_location

    def generate_tokens(self, encoder):
        self.tokens = encoder.encode(self.text)
        self.token_count = len(self.tokens)

    def to_dict(self):
        result = copy.copy(vars(self))
        result.pop("tokens", None)
        return result

    @staticmethod
    def from_dict(dictionary):
        dictionary.pop("is_limerick", None)
        limerick_text = dictionary.get("limerick", None)
        if limerick_text is not None:
            dictionary.pop("limerick", None)
            dictionary["text"] = limerick_text
        result = Limerick(**dictionary)
        return result


def read_and_init_limericks(file_path):
    result = []
    encoder = tiktoken.encoding_for_model("gpt-4")
    with open(file_path, "r") as file:
        limerick_dict_list = json.load(file)
        for limerick_dict in limerick_dict_list:
            limerick = Limerick.from_dict(limerick_dict)
            limerick.generate_tokens(encoder)
            result.append(limerick)
    return result


def select_limericks_to_answer(limerick_list, number_of_answers):
    selected_limerick_dict = {}
    while len(selected_limerick_dict) < number_of_answers:
        index = random.randint(0, len(limerick_list) - 1)
        limerick = limerick_list[index]
        if selected_limerick_dict.get(limerick.id, None) is None:
            selected_limerick_dict[limerick.id] = copy.copy(limerick)
    result = list(selected_limerick_dict.values())
    return result


def generate_answers(limerick_list, number_of_answers, file_path):
    selected_limerick_list = select_limericks_to_answer(limerick_list, number_of_answers)
    result = []
    for limerick in selected_limerick_list:
        print(limerick.text)
        print("Create a question for the limerick above")
        question = input()
        print("What is the answer to the question?")
        answer = input()
        limerick.question = question
        limerick.answer = answer
    result_dict_list = [limerick.to_dict() for limerick in selected_limerick_list]
    with open(file_path, "w") as file:
        json.dump(result_dict_list, file, indent=4)
    return result


if __name__ == '__main__':
    generate_answers(read_and_init_limericks(LIMERICK_DATASET_FILE), NUMBER_OF_TEST_QUESTIONS_TO_GENERATE, GENERATED_QUESTION_FILE)

