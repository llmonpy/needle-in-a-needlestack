# Needle in a Needlestack

> In Dolores Park, 'neath the sun's bright array,  
> I feasted on a sandwich, in the midday.  
> While Waldo, donned in cap, hid in plain sight,  
> Like a needle in a haystack, out of the light,  
> Unseen by me, focus led astray, I missed the play 

NIAN is a tool to test how well LLMs pay
attention in specific parts of their context window. It generates
tests with a large list of limericks. It places
a test limerick in a specified location in 
the prompt and asks the LLM to answer a question about
the test limerick.  A typical test might use 10 test
limericks placed at 10-20 locations in the prompt.  
It uses this database of limericks:

> https://zenodo.org/records/5722527



## Using NIAN
updated2
When you run the tool, it will give you a menu of 3
options:
1. Generate questions and answers for a limerick
2. Generate a test with limericks and test questions
3. Run a test

__full_questions.json__ contains the limericks with
questions and answers I have generated.  You can add to
the list by generating your own questions with option 1
and manually adding them to the file.

To generate test, you need to specify the length
of the prompts you want to generate.  You do that by
changing the values in the __PROMPT_SIZE_LIST__ array in
__main.py__.  The values in the array are the lengths of
the prompts.  This example:
> `PROMPT_SIZE_LIST = [ 1500, 15000, 100000]`

will generate prompts of 1500, 15000, and 100000 tokens.
Token counts are estimated using OpenAI's GPT-4 tokenizer.
The test data is stored in test_[length].json files.
The test file includes the number of questions set in:
> `NUMBER_OF_QUESTIONS_PER_PROMPT = 10`

When you run a test, it will use the test files implied by
the values in the __PROMPT_SIZE_LIST__ array. For example:
> `PROMPT_SIZE_LIST = [ 1500, 15000, 100000]`

will use the test files: test_1500.json, test_15000.json,
and test_100000.json.  The test will ask the LLM to answer
each question in the test file and record the answer. The
test limericks will be placed in the prompt at the locations
in __ROUGH_QUESTION_LOCATIONS__.  For example:
>ROUGH_QUESTION_LOCATIONS = [100, 1200, 5700, 11700, 80000, 99700]




todo:
- make running test use prompt_size_list
- make the default number of questions 10
- change name of LimerickPrompt to LimerickTest
- change the name of LimerickPrompk.text to prompt_introduction
- move question_locations into a test config that includes prompt size as well