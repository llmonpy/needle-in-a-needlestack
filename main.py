import copy
import json
import random

import tiktoken

PROMPT_SIZE_LIST = [ 1500, 6000, 12000, 30000]

ROUGH_QUESTION_LOCATIONS = [100, 1200, 5700, 11700, 29700]

#QUESTION_LOCATIONS = [100, 1300, 2800, 4300, 5800, 7300, 8800, 10300, 11800, 13300, 14800, 16300, 17800, 19300, 20800,
#                      22300, 23800, 25300, 26800, 28300, 29800]


class Limerick:
    def __init__(self, id, author, text, question=None, answer=None, tokens=None, token_count=None):
        self.id = id
        self.author = author
        self.text = text
        self.question = question
        self.answer = answer
        self.tokens = tokens
        self.token_count = token_count

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

class LimerickPrompt:
    def __init__(self, limerick, question, answer, prompt, prompt_size):
        self.limerick = limerick
        self.question = question
        self.answer = answer
        self.prompt = prompt
        self.prompt_size = prompt_size

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dictionary):
        result = LimerickPrompt(**dictionary)
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


def find_the_longest_limerick_in_tokens(limerick_list):
    max_token_count = 0
    max_limerick = None
    for limerick in limerick_list:
        if limerick.token_count > max_token_count:
            max_token_count = limerick.token_count
            max_limerick = limerick
    print("Max length:", max_token_count)
    print(max_limerick.text)
    result_dict = { "max_token_count": max_token_count, "max_limerick": max_limerick.to_dict() }
    with open("max_limerick.json", "w") as file:
        json.dump(result_dict, file, indent=4)
    return max_limerick


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


def select_questions_for_prompt(file_path, number_of_questions):
    with open(file_path, "r") as file:
        question_dict_list = json.load(file)
    question_list = [Limerick.from_dict(question_dict) for question_dict in question_dict_list]
    question_dict_list = []
    for question in question_list:
        if question.token_count is None:
            question.generate_tokens()
        question_dict_list.append(question.to_dict())
    with open("full_questions.json", "w") as file:
        json.dump(question_dict_list, file, indent=4)
    selected_question_dict = {}
    while len(selected_question_dict) < number_of_questions:
        index = random.randint(0, len(question_list) - 1)
        question = question_list[index]
        if selected_question_dict.get(question.id, None) is None:
            selected_question_dict[question.id] = question
    result = list(selected_question_dict.values())
    return result


def select_limericks_for_prompt(limerick_list, question_dict, max_token_count):
    current_token_count = 0
    selected_limerick_dict = {}
    while current_token_count < max_token_count:
        index = random.randint(0, len(limerick_list) - 1)
        limerick = limerick_list[index]
        if selected_limerick_dict.get(limerick.id, None) is None:
            selected_limerick_dict[limerick.id] = limerick
            current_token_count += limerick.token_count
    result = list(selected_limerick_dict.values())
    return result


def generate_prompts(limerick_list, prompt_size_list, rough_question_location_list):
    prompts = []
    selected_question_list = select_questions_for_prompt("full_questions.json", len(rough_question_location_list))
    max_token_count = prompt_size_list[-1]
    selected_limerick_list = select_limericks_for_prompt(limerick_list, max_token_count)
    for prompt_size in prompt_size_list:
        prompt = generate_prompt_of_size(prompt_size, selected_limerick_list, selected_question_list,
                                         rough_question_location_list)
        prompts.append(LimerickPrompt(selected_limerick_list, selected_question_list, prompt, prompt_size))
    return prompts


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # find_the_longest_limerick_in_tokens(read_and_init_limericks("limerick_dataset_oedilf_v3.json"))
    print("Enter 1 to generate questions, 2 to generate prompts or 3 to run prompts:")
    user_input = input()
    user_input = user_input.strip()
    if user_input == "1":
        generate_answers(read_and_init_limericks("limerick_dataset_oedilf_v3.json"), 7, "questions2.json")
    elif user_input == "2":
        generate_prompts(read_and_init_limericks("limerick_dataset_oedilf_v3.json"), PROMPT_SIZE_LIST, ROUGH_QUESTION_LOCATIONS)
    elif user_input == "3":
        pass
    else:
        print("Invalid input")
