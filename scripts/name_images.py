import argparse
import base64
import csv
from openai import OpenAI, RateLimitError
from pathlib import Path
import pandas as pd
import time
from typing import Any, Dict, List
from tqdm import tqdm
import random
import yaml

SEED = 10
random.seed(SEED)

client = OpenAI()

def get_completions(preamble: str, prompt: str, 
                    base64_image: str, n: int=5) -> List[Any]:
  """
  Gets chat completions using the OpenAI API, returns a list of responses.
  """
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    seed=SEED,
    n=n,
    temperature=1,
    messages=[
      # Preamble
      {"role": "system", 
      "content": [
          {"type": "text",
            "text":preamble
          }
      ]
      },
      # Request
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
              "detail": "low"
            },
          },
        ],
      }
    ],
    max_tokens=100,
  )

  return response.choices

def encode_image(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_responses(preamble: str, prompt: str, encoded_image: str, n=5) -> List[Any]:
  """
  Gets a series of n completions from the prompt, preamble and encoded image.
  """
  completions = get_completions(preamble, prompt, encoded_image, n)
  responses = []
  for completion in completions:
    responses.append(completion.message.content)

  return responses

def test_all_paths(preamble: str, prompt_format: str, stimuli_paths: List[str], 
                   matches: bool, response_writer: Any, subordinate_names: Dict[str, List[str]]) -> List[str]:

  failed_paths = []

  for stim_path in tqdm(stimuli_paths):
    stim_path = Path(stim_path.strip())
    stim_name = stim_path.stem
    basic_category_name = stim_path.parents[0].stem.lower()
    if prompt_type != "basic" and prompt_type != "subordinate":
      prompt = prompt_format
    else:
      prompt = format_truefalse_prompt(prompt_format, matches, stim_name, basic_category_name, subordinate_names)
    encoded_image = encode_image(stim_path)

    try:
      responses = ['hi'] * num_completions # dummy response array
      #responses = get_responses(preamble, prompt, encoded_image, num_completions)
    except RateLimitError as e:
      if "RPD" in e.message: # stop if Requests Per Day has been reached
        print("Requests Per Day reached. Stopping! Stimuli untested:")
        print(failed_paths)
        exit(1)
      failed_paths.append(str(stim_path))
      time.sleep(2) # if rate limit is due to tokens per minute, sometimes it helps to just wait
    else:
      line = [preamble, prompt, basic_category_name, stim_name] + responses
      response_writer.writerow(line)

  return failed_paths

def format_truefalse_prompt(prompt_format: str, query_matches_gold: bool, stim_name: str, 
                            basic_category_name: str, subordinate_names: Dict[str, List[str]]) -> str:

  if prompt_type == "basic":
    if query_matches_gold:
      prompt_category = basic_category_name
    else:
      if basic_category_name == "bird":
        prompt_category = "flower"
      elif basic_category_name == "dog":
        prompt_category = "tree"
      elif basic_category_name == "flower":
        prompt_category = "dog"
      elif basic_category_name == "tree":
        prompt_category = "bird"
  elif prompt_type == "subordinate":
    stim_subordinate = stim_name.split("_")[:-1]
    stim_subordinate = " ".join(stim_subordinate).lower()
    alternatives = [x for x in subordinate_names[basic_category_name] if x != stim_subordinate]
    if query_matches_gold:
      prompt_category = stim_subordinate
    else:
      prompt_category = alternatives[2]
      
    if basic_category_name == "tree" and prompt_type == "subordinate":
      prompt_category += " tree"

  if prompt_category[0] in ["a","e","i","o","u"]:
    indefinite = "an"
  else:
    indefinite = "a"
  prompt = prompt_format.format(indefinite, prompt_category)
  return prompt

def add_comparison_to_gold_labels(responses):
    basic_cat = responses['category'].apply(lambda x: x.lower())
    sub_cat = responses['stimulus'].apply(lambda x: x.split("_")[0].replace("_"," ").lower())
    x = pd.merge(basic_cat, sub_cat, right_index=True, left_index=True)

    cleaned_responses = responses['response'].apply(lambda x: x.replace('.','').lower())
    x = x.merge(cleaned_responses, right_index=True, left_index=True)

    responses['is_subordinate'] = x.apply(lambda x: x.stimulus in x.response, axis=1)
    responses['contains_basic'] = x.apply(lambda x: x.category in x.response,axis=1)
    responses['is_basic'] =  responses['contains_basic'] & ~responses['is_subordinate']
    return responses

def reformat_file(query_matches_gold, preamble_type, prompt_type, save_file_name):
  """
  Reformats file from wide-format into long format for data analysis, 
  and adds automatically-labelled category type labels (basic/subordinate).
  """
  responses = pd.read_csv(save_file_name, delimiter='\t')
  responses = responses.melt(
    id_vars=['category', 'stimulus', 'preamble', 'prompt'],var_name='response_no',
    value_name='response'
  )
  if prompt_type == "basic" or prompt_type == "subordinate":
    cleaned_responses = responses['response'].apply(lambda x: str(x).replace(".","").lower())
    responses['matches'] = cleaned_responses.apply(lambda x: str(query_matches_gold).lower() in x)
  else:
    responses = add_comparison_to_gold_labels(responses)
  responses['preamble_type'] = preamble_type
  responses['prompt_type'] = prompt_type
  responses.to_csv(save_file_name, index=False)

        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('system_prompt_type', help="Name of system prompt type. See system_prompts.yaml")
  parser.add_argument('stimuli_type')
  parser.add_argument('--basic', '-b', action='store_true', help="Whether or not to use basic T/F prompt") # 
  parser.add_argument('--subordinate', '-s', action='store_true', help="Whether or not to use subordinate T/F prompt")
  parser.add_argument('--matches', '-m', action='store_true', help="Whether or not queries should match gold labels using T/F prompt") # 
  parser.add_argument('--reformat', '-r', action='store_true', help="Whether or not to reformat output file to longer form") #
  parser.add_argument('-n', default=10, help="Number of completions to get") #  
  args = parser.parse_args()

  preamble_type = args.system_prompt_type
  if args.basic:
    prompt_type = 'basic'
  elif args.subordinate:
    prompt_type = 'subordinate'
  else:
    prompt_type = 'default'
  stimuli = args.stimuli_type
  query_matches_gold = args.matches # in a T/F question, whether the query should match the gold label
  reformat = args.reformat
  num_completions = args.n

  if prompt_type == "basic" or prompt_type == "subordinate":
    prompt = "Does this image contain {} {}? Answer true or false."
  else:
    prompt = "What's in this image? Answer as quickly as possible using one or two words."

  scripts_dir = Path(__file__).parent.resolve()
  project_dir = Path(__file__).parent.parent.resolve()

  with open( scripts_dir / "system_prompts.yaml", 'r') as f:
    preambles = yaml.load(f, yaml.CLoader)
  preamble = preambles[preamble_type]
  
  with open(scripts_dir / "subordinate_names.yaml", 'r') as f:
    subordinate_names = yaml.load(f, yaml.CLoader)

  Path( project_dir / f'results/').mkdir(parents=True, exist_ok=True) # create the results/ directory if needed
  save_file_name = project_dir / f'results/{preamble_type}_{prompt_type}_{stimuli}.csv'
  stimuli_list_file = scripts_dir / "stimuli_to_test.txt"

  with open(stimuli_list_file, 'r') as stim_file:
    stimuli_paths = stim_file.readlines()
  
  with open(save_file_name, 'a', newline='') as csvfile:
    response_writer = csv.writer(csvfile, delimiter='\t')
    response_writer.writerow(['preamble', 'prompt', 'category', 'stimulus'] + [f'response_{x}' for x in range(num_completions)])

    failed_paths = test_all_paths(preamble, prompt, stimuli_paths, query_matches_gold, response_writer, subordinate_names)
    
    # Sometimes due to rate limiting we have to try again!
    while failed_paths:
      print(failed_paths)
      failed_paths = test_all_paths(preamble, prompt, failed_paths, query_matches_gold, response_writer, subordinate_names)

  if reformat:
    reformat_file(query_matches_gold, preamble_type, prompt_type, save_file_name)
