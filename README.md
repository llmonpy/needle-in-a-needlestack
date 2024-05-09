# Needle in a Needlestack

> In Dolores Park, 'neath the sun's bright array,  
> I feasted on a sandwich, in the midday.  
> While Waldo, donned in cap, hid in plain sight,  
> Like a needle in a haystack, out of the light,  
> Unseen by me, focus led astray, I missed the play 
> 

Needle in a haystack (NIAH) has been a wildly popular test for evaluating how effectively LLMs can pay attention to 
the content in their context window.  As LLMs have improved NIAH has become too easy.  Needle in a Needlestack (NIAN)
is a new, more challenging benchmark.  You can see initial results on the [NIAN website](https://nian.llmonpy.ai).

NIAN creates a list of limericks from a large [database of limericks](https://zenodo.org/records/5722527) and asks a 
question about a specific limerick that have been placed at a test location. Each test will typically use 5 to 10
test limericks placed at 5 to 10 locations in the prompt.  Each test is repeated 2-10 times.  It is amazing that an 
LLM can answer these questions at all! Here is a [link to an example prompt](artifacts/sample_prompt.txt).  The
question is "Does Mr. Thistle follow our rules?" and the associated limerick is in the middle of the prompt:

> Mr. Thistle, what's this all about?  
> All our rules you dismiss — baldly flout.  
> I have read your epistle;  
> It made my hairs bristle.  
> I'm blowing the whistle — you're out!

Evaluating the LLM responses is always challenging and NIAN is worthless without accurate evaluation. To get more 
accurate evaluation, NIAN uses 5 LLMs to evaluate the responses and pass/fail is determined by majority vote.  NIAN 
includes tools to evaluate the evaluators and improve them with few shot prompting.

Given the number of trials and the 5 LLM calls per trial, it is important that NIAN make many LLM calls in parallel.
It uses a rate limiter (that will become its own package) to manage the rate of LLM calls.  NIAN is quite fast.  For
example, it can make 600 LLM calls to evaluate "open-mistral-7b" with 5 questions at 5 locations, 5 times in 35 seconds
on my mac. 

## Running your own tests
NIAN supports the LLMs I have access too -- OpenAI, Anthropic and Mistral.  Adding new LLMs is easy, and it will be
covered later in this document.  You can set the rate limits for each LLM, but you will want generous rate limits to
run the tests.  My development was done with OpenAI Tier 4, Anthropic Tier 4 and Mistral Tier 2.  NIAN looks for the
standard API keys -- OPENAI_API_KEY, ANTHROPIC_API_KEY, and MISTRAL_API_KEY.  It looks first for these keys prefixed 
with "NIAN_" if you want to use NIAN specific keys.

## test_config.py
You configure a test by setting the CURRENT_TEST_CONFIG or by changing DEFAULT_TEST_CONFIG. A TestConfig looks like this:

```
DEFAULT_TEST_CONFIG = TestConfig(model_list=[MISTRAL_7B, GPT3_5],
                                 test_thread_count=100,
                                 evaluator_model_list=EVALUATOR_MODEL_LIST,
                                 default_evaluator=DefaultEvaluator(EVALUATOR_MODEL_LIST),
                                 number_of_questions_per_trial=5,
                                 repeat_question_limerick_count=100,
                                 trial_count=5,
                                 location_count=10)
```
Most of the settings are self-explanatory.  The model_list is the list of LLMs to use for the test.  The 
test_thread_count is the number of threads to used to generate answers etc.  Repeat_question_limerick_count is the
is least obvious.  It controls how many times the limerick that the prompt asks a question about is repeated.  When it
is repeated, it is repeated in the prompt at the same location.  This is a way to test if repeating information in the
prompt helps the LLM answer the question.  The trial_count is the number of trials to run for each question and each 
location.  The total number of times a prompt will be sent to each model tested is:

> `number_of_questions_per_trial * trial_count * location_count `


## Adding new LLMs
Adding a new LLM is easy.  You need to create a new class that extends LLMClient.  You need to implement the constructor
and the do_prompt method.  The client will look something like this.

```
class OpenAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thead_pool=None):
        super().__init__(model_name, max_input, rate_limiter, thead_pool)
        key = get_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text."):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result

```

You will also need to create a global variable in llm_clients.py to make the model accessible. The client will need
a rate limiter and a thread_pool to use for evaluation.  The rate limiter is a RateLimiter object that will be used to
manage the rate of LLM calls:

```
MISTRAL_RATE_LIMITER = RateLlmiter(*spread_requests(1000)) #used minute spread to seconds because tokens are TPM not TPS
MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 24000, MISTRAL_RATE_LIMITER, MISTRAL_EXECUTOR)
```

## Tools
NIAN includes tools to generate questions and answers for limericks, vet that LLMs can answer the questions,
identify unique passing and failing answers to questions, report on dissent among LLMs on answer evaluation, report
on variation in LLM answers when it is asked the same question multiple times at the same location, a tool to 
create plots from existing tests and a tool to run tests.  All the scripts are in bin and settup the venv for you.


### Nian
Nian runs the tests as configured in test_config.py.  It presents 

Tools
    running tests
    generating questions
    evaluating evaluators

Adding new llm clients


Adding and evaluating new questions

Improving plots


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