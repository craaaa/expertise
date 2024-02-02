# About
This repository contains code for the paper Prompting invokes expert-like downward shifts in GPT-4V’s conceptual hierarchies (Leong and Lake, 2024).

## Data
Data were obtained from [van Hoef (2022)](https://osf.io/vdka2/?view_only=ac6c78793a354a1f95da6c36b5ac6163) and [Parkhi et al. (2012)](https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset?resource=download). We extract a subset of data from the Cats And Dogs dataset.

## Replication
First, obtain a list of stimuli files to be tested. The stimuli files should be in `data/`. Running `scripts/list_files.py` should create a file `scripts/stimuli_to_test.txt` containing all files to be tested.

```
python list_files.py {dataset_name}
python list_files.py van_hoef
```

Then, run `scripts/name_images.py` to use the OpenAI API to name images. Results will be saved in `results/`.

```
python name_images.py {system_prompt} {stimuli_type} [--basic] [--subordinate] [--matches] [--reformat] [-n N]
```

Valid options for `system_prompt` are available in `scripts/system_prompts.yaml`.
Options include: 
- `default`
- `novice`
- `dogexpert`
- `dog_explicit`
- `zoologist`
- ...

`stimuli_type` is a string that specifies the types of stimuli tested and is useful for additional notes.

Flags:
- `--basic`, `-b`: use this flag to get basic T/F queries 
- `--subordinate` `-s`: use this flag to get subordinate T/F queries
- `--matches`, `-m`: use this flag to get T/F queries that match the gold label
- `--reformat`, `-r`: use this flag to reformat the results file into long format

Example use to name all images using the `default`/`dogexpert` system prompt:
```
python name_images.py default all -r
python name_images.py dogexpert all -r
```

Example use to query basic labels using the True/False prompt:
```
python name_images.py default all -b -r
```

## Analysis
Use `analysis.Rmd` to obtain all graphs and statistics used in the paper.
