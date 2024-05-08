import csv
import json
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat

import fire
import pyrootutils
import yaml
from minicons import scorer
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def score(
    preamble: str,
    prompt_type: str,
    prompt_format: str,
    stimuli_paths: list[str],
    subordinate_names: dict[str, list[str]],
    query_matches_gold: bool,
    alt_n: int,
):
    lm = scorer.VLMScorer(
        "llava-hf/llava-1.5-7b-hf",
        device="cuda:0",
        # attn_implementation="flash_attention_2",
    )

    def get_sub_name(filename):
        split = str(filename.stem).rsplit("_", 1)[0]
        return split.replace("_", " ")

    def get_indefinite(noun: str):
        if noun[0] in ["a", "e", "i", "o", "u"]:
            indefinite = "an"
        else:
            indefinite = "a"
        return indefinite

    def get_tf_queries(
        query_matches_gold, prompt_type, prompt_format, sub_cat, basic_cat
    ):
        if prompt_type == "basic":
            correct_prompt = basic_cat
            basic_cats = [
                x for x in ["bird", "flower", "tree", "dog"] if x != basic_cat
            ]
            alt_prompt = basic_cats[alt_n]
        elif prompt_type == "subordinate":
            alternatives = [x for x in subordinate_names[basic_cat] if x != sub_cat]
            correct_prompt = sub_cat
            alt_prompt = alternatives[alt_n]

        correct_prompt = prompt_format.format(
            get_indefinite(correct_prompt), correct_prompt
        )
        alt_prompt = prompt_format.format(get_indefinite(alt_prompt), alt_prompt)

        if query_matches_gold:
            return get_queries(correct_prompt, "True", "False")
        else:
            return get_queries(alt_prompt, "False", "True")

    def get_queries(prompt_format, first_answer, second_answer):
        query_base = f"{preamble} USER: <image>\n{prompt_format} ASSISTANT: "
        return [
            query_base + first_answer.capitalize(),
            query_base + second_answer.capitalize(),
        ]

    stimuli_paths = [Path(stim_path.strip()) for stim_path in stimuli_paths]
    basic_category_names = [
        stim_path.parents[0].stem.lower() for stim_path in stimuli_paths
    ]
    subordinate_category_names = [
        get_sub_name(stim_path) for stim_path in stimuli_paths
    ]

    lines = []
    for b, s, img_path in tqdm(
        zip(basic_category_names, subordinate_category_names, stimuli_paths),
        total=len(basic_category_names),
    ):
        if prompt_type != "basic" and prompt_type != "subordinate":
            text_batch = get_queries(
                prompt_format=prompt_format, first_answer=s, second_answer=b
            )
        else:
            text_batch = get_tf_queries(
                query_matches_gold=query_matches_gold,
                prompt_type=prompt_type,
                prompt_format=prompt_format,
                sub_cat=s,
                basic_cat=b,
            )
        img = Image.open(img_path)
        first_score, second_score = lm.sequence_score(
            text_batch=text_batch, image_batch=[img] * 2
        )
        lines.append(
            [
                preamble,
                b,
                s,
                text_batch[0],
                text_batch[1],
                first_score,
                second_score,
                first_score > second_score,
            ]
        )
    return lines


def main(
    output_dir: str = PROJECT_ROOT / "results",
    preamble_type: str = "default",
    basic: bool = False,
    subordinate: bool = False,
    match: bool = True,
    alt_n: int = 0,  # alternative index number for mismatched T/F questions
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
        prompt = "What's in this image? Answer as \
        quickly as possible using one or two words."
    elif prompt_type == "basic" or prompt_type == "subordinate":
        prompt = "Does this image contain {} {}? Answer true or false."

    with open(PROJECT_ROOT / "scripts" / "system_prompts.yaml", "r") as f:
        preambles = yaml.load(f, yaml.CLoader)
    preamble = preambles[preamble_type]

    with open(PROJECT_ROOT / "scripts" / "subordinate_names.yaml", "r") as f:
        subordinate_names = yaml.load(f, yaml.CLoader)

    # create the results/ directory if needed
    project_dir.mkdir(parents=True, exist_ok=True)
    save_file_name = project_dir / f"{preamble_type}_{prompt_type}_scores.csv"
    stimuli_list_file = PROJECT_ROOT / "scripts" / "stimuli_to_test.txt"

    with open(stimuli_list_file, "r") as stim_file:
        stimuli_paths = stim_file.readlines()

    query_params = {
        "preamble": preamble,
        "preamble_type": preamble_type,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "prompt_format": prompt,
        "query_matches_gold": query_matches_gold,
        "alt_n": alt_n,
    }

    print(f"Config: {pformat(query_params)}")

    with open(project_dir / "query_params.json", "w") as f:
        json.dump(query_params, f)

    scores = score(
        preamble=preamble,
        prompt_type=prompt_type,
        prompt_format=prompt,
        stimuli_paths=stimuli_paths,
        subordinate_names=subordinate_names,
        query_matches_gold=query_matches_gold,
        alt_n=alt_n,
    )
    with open(save_file_name, "a", newline="") as csvfile:
        response_writer = csv.writer(csvfile, delimiter="\t")

        response_writer.writerow(
            [
                "preamble",
                "category",
                "stimulus",
                "first_prompt",
                "second_prompt",
                "first_score",
                "second_score",
                "prefer_first",
            ]
        )
        for row in scores:
            response_writer.writerow(row)


if __name__ == "__main__":
    fire.Fire(main)
