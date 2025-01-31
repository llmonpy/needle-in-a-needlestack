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
import random
# import tiktoken

NUMBER_OF_TEST_QUESTIONS_TO_GENERATE = 5
GENERATED_QUESTION_FILE = "questions.json"
FULL_QUESTION_FILE = "full_questions.json"
LIMERICK_DATASET_FILE = "limerick_dataset_oedilf_v3.json"
AVERAGE_TEXT_TO_GPT4_TOKEN_RATIO = 1.4813021426742123 # calculated from the limerick dataset and tiktoken

#token_ratio_list = [] # used to calculate the average token ratio of the limerick dataset

class Limerick:
    def __init__(self, id, author, text, question=None, answer=None, tokens=None, token_count=None, target_location=0,
                 question_vetted=False, alternate_answers=None):
        self.id = id
        self.author = author
        self.text = text
        self.question = question
        self.answer = answer
        self.alternate_answers = alternate_answers
        self.tokens = tokens
        self.token_count = token_count
        self.target_location = target_location
        self.question_vetted = question_vetted

    def generate_tokens(self, encoder):
        if encoder is None:
            self.tokens = []
            word_count = len(self.text.split())
            self.token_count = word_count * AVERAGE_TEXT_TO_GPT4_TOKEN_RATIO
        elif self.text is None or len(self.text) == 0 or len(self.text.split()) == 0:
            self.tokens = []
            self.token_count = 0
        else:
            # this is only used to calculate AVERAGE_TEXT_TO_GPT4_TOKEN_RATIO, don't want to import tiktoken
            raise NotImplementedError("This method is not implemented yet")
            #self.tokens = encoder.encode(self.text)
            word_count = len(self.text.split())
            #self.token_count = len(self.tokens)
            self.token_count = word_count * AVERAGE_TEXT_TO_GPT4_TOKEN_RATIO
            #token_ratio_list.append(self.token_count/word_count)

    def has_alternate_answers(self):
        result = self.alternate_answers is not None and len(self.alternate_answers) > 0
        return result

    def get_all_answers(self):
        result = [self.answer]
        if self.has_alternate_answers():
            result.extend(self.alternate_answers)
        return result

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
    #encoder = tiktoken.encoding_for_model("gpt-4")
    with open(file_path, "r") as file:
        limerick_dict_list = json.load(file)
        for limerick_dict in limerick_dict_list:
            limerick = Limerick.from_dict(limerick_dict)
            limerick.generate_tokens(None)
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
    #generate_answers(read_and_init_limericks(LIMERICK_DATASET_FILE), NUMBER_OF_TEST_QUESTIONS_TO_GENERATE, GENERATED_QUESTION_FILE)
    read_and_init_limericks(LIMERICK_DATASET_FILE)
    #print("Average token ratio: "+str(sum(token_ratio_list)/len(token_ratio_list)))
