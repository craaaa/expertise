import base64
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List

import fire
import pandas as pd
import pyrootutils
import regex as re
import torch
import yaml
from dotenv import load_dotenv
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava_utils import get_input_ids, get_prompt, process_image
from openai import OpenAI, RateLimitError
from tqdm import tqdm

SEED = 10
random.seed(SEED)
load_dotenv()

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

client = OpenAI()


def openai_get_completions(
    preamble: str,
    prompt: str,
    base64_image: str,
    num_completions: int,
    max_tokens: int,
    temperature: float,
) -> List[Any]:
    """
    Gets chat completions using the OpenAI API, returns a list of responses.
    """
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        seed=SEED,
        n=num_completions,
        temperature=temperature,
        messages=[
            # Preamble
            {"role": "system", "content": [{"type": "text", "text": preamble}]},
            # Request
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
    )

    return response.choices


def openai_encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def openai_get_responses(
    preamble: str,
    prompt: str,
    encoded_image: str,
    num_completions: int,
    temperature: float,
    max_tokens: int,
) -> List[Any]:
    """
    Gets a series of n completions from the prompt, preamble and encoded image.
    """
    completions = openai_get_completions(
        preamble=preamble,
        prompt=prompt,
        encoded_image=encoded_image,
        num_completions=num_completions,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    responses = []
    for completion in completions:
        responses.append(completion.message.content)

    return responses


def test_all_paths(
    test_model: str,
    preamble: str,
    prompt_type: str,
    prompt_format: str,
    stimuli_paths: List[str],
    matches: bool,
    response_writer: Any,
    subordinate_names: Dict[str, List[str]],
    num_completions: int,
    top_p: int,
    num_beams: int,
    max_tokens: int,
    temperature: float,
    sep: str,
) -> List[str]:
    failed_paths = []

    if test_model == "llava":
        disable_torch_init()
        model_path = "liuhaotian/llava-v1.5-7b"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
        )

    for stim_path in tqdm(stimuli_paths):
        stim_path = Path(stim_path.strip())
        stim_name = stim_path.stem
        basic_category_name = stim_path.parents[0].stem.lower()
        if prompt_type != "basic" and prompt_type != "subordinate":
            prompt = prompt_format
        else:
            prompt = format_truefalse_prompt(
                prompt_format=prompt_format,
                prompt_type=prompt_type,
                query_matches_gold=matches,
                stim_name=stim_name,
                basic_category_name=basic_category_name,
                subordinate_names=subordinate_names,
            )

        if test_model == "openai":
            encoded_image = openai_encode_image(stim_path)
            try:
                responses = ["hi"] * num_completions  # dummy response array
                responses = openai_get_responses(
                    preamble=preamble,
                    prompt=prompt,
                    encoded_image=encoded_image,
                    num_completions=num_completions,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except RateLimitError as e:
                if "RPD" in e.message:  # stop if Requests Per Day has been reached
                    print("Requests Per Day reached. Stopping! Stimuli untested:")
                    print(failed_paths)
                    exit(1)
                failed_paths.append(str(stim_path))
                time.sleep(2)  # if rate limit is due to tokens per minute, just wait
            else:
                line = [preamble, prompt, basic_category_name, stim_name] + responses
                response_writer.writerow(line)

        elif test_model == "llava":
            image_file = str(stim_path)
            prompt_text = get_prompt(
                query=prompt,
                model=model,
                model_name=get_model_name_from_path(model_path),
                system_prompt=preamble,
                explicit_conv_mode=None,
            )

            image_sizes, images_tensor = process_image(
                image_file=image_file, image_processor=image_processor, model=model
            )
            input_ids = get_input_ids(prompt_text, tokenizer)

            responses = []
            for i in range(num_completions):
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=images_tensor,
                        image_sizes=image_sizes,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                    0
                ].strip()
                responses.append(outputs)
            line = [preamble, prompt, basic_category_name, stim_name] + responses
            response_writer.writerow(line)

    return failed_paths


def format_truefalse_prompt(
    prompt_format: str,
    prompt_type: str,
    query_matches_gold: bool,
    stim_name: str,
    basic_category_name: str,
    subordinate_names: Dict[str, List[str]],
) -> str:
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
        alternatives = [
            x for x in subordinate_names[basic_category_name] if x != stim_subordinate
        ]
        if query_matches_gold:
            prompt_category = stim_subordinate
        else:
            prompt_category = alternatives[2]

        if basic_category_name == "tree" and prompt_type == "subordinate":
            prompt_category += " tree"

    if prompt_category[0] in ["a", "e", "i", "o", "u"]:
        indefinite = "an"
    else:
        indefinite = "a"
    prompt = prompt_format.format(indefinite, prompt_category)
    return prompt


def add_comparison_to_gold_labels(responses):
    basic_cat = responses.category
    sub_cat = responses.stimulus.str.rsplit("_", n=1).str[0]
    comparison_df = pd.DataFrame(
        {"basic": basic_cat, "subordinate": sub_cat, "actual": responses.response}
    )

    def compare(gold: str, actual: str):
        gold = clean_answer(gold)
        actual = clean_answer(actual)
        if gold == actual:
            return True
        if gold in actual:
            return True
        if actual in gold:
            return True
        return False

    responses["is_subordinate"] = comparison_df.apply(
        lambda x: compare(gold=x.subordinate, actual=x.actual)
    )
    responses["contains_basic"] = comparison_df.apply(
        lambda x: compare(gold=x.basic, actual=x.actual)
    )
    responses["is_basic"] = responses["contains_basic"] & ~responses["is_subordinate"]
    return responses


def clean_answer(original: str):
    return re.sub("[^a-zA-Z ]+", "", original).lower()


def reformat_file(query_matches_gold, preamble_type, prompt_type, save_file_name):
    """
    Reformats file from wide-format into long format for data analysis,
    and adds automatically-labelled category type labels (basic/subordinate).
    """

    def check_match(gold: bool, actual: str):
        actual = clean_answer(actual)
        if gold:
            return actual in ["yes", "true"]
        else:
            return actual in ["no", "false"]

    responses = pd.read_csv(save_file_name, delimiter="\t")
    responses = responses.melt(
        id_vars=["category", "stimulus", "preamble", "prompt"],
        var_name="response_no",
        value_name="response",
    )
    if prompt_type == "basic" or prompt_type == "subordinate":
        gold = responses.apply(lambda x: x.category in x.prompt, axis=1)
        comparison_df = pd.DataFrame({"gold": gold, "actual": responses.response})

        responses["matches"] = comparison_df.apply(
            lambda x: check_match(x.gold, x.actual), axis=1
        )

    else:
        responses = add_comparison_to_gold_labels(responses)
    responses["preamble_type"] = preamble_type
    responses["prompt_type"] = prompt_type
    responses.to_csv(save_file_name, index=False)


def main(
    output_dir: str = PROJECT_ROOT / "results",
    preamble_type: str = "default",
    stimuli_type: str | None = None,
    basic: bool = False,
    subordinate: bool = False,
    match: bool = True,
    reformat: bool = True,
    num_completions: int = 10,
    sep: str = ",",
    temperature: float = 1.0,
    top_p: float | None = None,
    num_beams: int = 1,
    max_new_tokens: int = 100,
    test_model: str = "llava",
):
    # create project directory inside output_dir based on the timestamp
    # plus a two-character random string
    project_dir = (
        output_dir
        / f"{datetime.today().strftime('%Y-%m-%d-%H%M%S')}_{os.urandom(2).hex()}"
    )
    if not output_dir.exists():
        output_dir.mkdir()
    if not project_dir.exists():
        project_dir.mkdir()

    prompt_type = "fastest"
    if basic:
        prompt_type = "basic"
    elif subordinate:
        prompt_type = "subordinate"

    # in a T/F question, whether the query should match the gold label
    query_matches_gold = match

    if prompt_type == "fastest":
        prompt = "What's in this image? Answer as quickly as possible using one or two words."  # noqa: E501
    elif prompt_type == "basic" or prompt_type == "subordinate":
        prompt = "Does this image contain {} {}? Answer true or false."

    with open(PROJECT_ROOT / "scripts" / "system_prompts.yaml", "r") as f:
        preambles = yaml.load(f, yaml.CLoader)
    preamble = preambles[preamble_type]

    with open(PROJECT_ROOT / "scripts" / "subordinate_names.yaml", "r") as f:
        subordinate_names = yaml.load(f, yaml.CLoader)

    # create the results/ directory if needed
    project_dir.mkdir(parents=True, exist_ok=True)
    save_file_name = project_dir / f"{preamble_type}_{prompt_type}_{test_model}.csv"
    stimuli_list_file = PROJECT_ROOT / "scripts" / "stimuli_to_test.txt"

    with open(stimuli_list_file, "r") as stim_file:
        stimuli_paths = stim_file.readlines()

    query_params = {
        "test_model": test_model,
        "preamble": preamble,
        "preamble_type": preamble_type,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "prompt_format": prompt,
        "query_matches_gold": query_matches_gold,
        "num_completions": num_completions,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }

    print(f"Config: {pformat(query_params)}")

    with open(project_dir / "query_params.json", "w") as f:
        json.dump(query_params, f)

    with open(save_file_name, "a", newline="") as csvfile:
        response_writer = csv.writer(csvfile, delimiter="\t")
        response_writer.writerow(
            ["preamble", "prompt", "category", "stimulus"]
            + [f"response_{x}" for x in range(num_completions)]
        )

        # Sometimes due to rate limiting we have to try multiple times!
        failed_paths = None
        while failed_paths is None or len(failed_paths) > 0:
            failed_paths = test_all_paths(
                test_model=test_model,
                preamble=preamble,
                prompt_type=prompt_type,
                prompt_format=prompt,
                stimuli_paths=stimuli_paths,
                response_writer=response_writer,
                subordinate_names=subordinate_names,
                num_completions=num_completions,
                sep=sep,
                top_p=top_p,
                max_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                matches=query_matches_gold,
            )

    if reformat:
        reformat_file(query_matches_gold, preamble_type, prompt_type, save_file_name)


if __name__ == "__main__":
    fire.Fire(main)
