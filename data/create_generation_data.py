import pandas as pd
import click
import os
from transformers import pipeline, set_seed, AutoTokenizer
from tqdm import tqdm
import warnings


# If you want to use a model which cannot take device_map="auto" and must be put onto a cuda device
# then add a substring of it here
NO_DEVICE_MAP_MODEL_SUBSTRINGS = ["MariumMT", "roberta"]
CUDA_DEVICE_INDEX = 0
N_GENS_MAX = 10


@click.command()
@click.option('--model_name_or_path', type=str, help="Name of the model to use for generation. "
                                                     "Should be able to accept the 'prompt' column of "
                                                     "the dataset with no changes")
@click.option('--data_dir', type=str, help="Path to folder in task_data, must have [train, validation, test].csv files")
@click.option('--task', default="text-generation", type=str, help="HuggingFace pipeline task of the model provided. "
                                                                  "Defaults to text-generation. "
                                                                  "CommonGen uses text2textgeneration. ")
@click.option('--constraint_max_tokens', default=None, type=int, help="For constraint datasets, "
                                                                      "specify the max words in prompt")
@click.option("--max_points", default=None, type=int, help="If specified the first max_points from each file are taken")
@click.option("--max_new_tokens", default=20, type=int, help="max new tokens to generate")
@click.option("--n_gens", default=3, type=int, help="Generate n_gens samples per prompt.")
@click.option('--random_seed', default=42, type=int, help="Random Seed Value")
def main(model_name_or_path, data_dir, task, constraint_max_tokens, max_points, max_new_tokens, n_gens, random_seed):
    if not n_gens < N_GENS_MAX:
        raise ValueError(f"To prevent likely failures {n_gens} must be less than {N_GENS_MAX}. "
                         f"Can change this if confident it wont cause issues. ")
    set_seed(random_seed)
    if "constraint_data" in data_dir:
        if constraint_max_tokens is None:
            warnings.warn(f"Constraint Dataset {data_dir} passed in without setting constraint_max_tokens...")
    else:
        assert constraint_max_tokens is None, f"Remove this at your own risk, it might overwrite the prompt column " \
                                              f"for task datasets"

    output_path = data_dir.replace("task_data", "generated_data").replace("constraint_data", "generated_data")
    model_name = model_name_or_path.split("/")[-1]
    os.makedirs(output_path+f"/{model_name}", exist_ok=True)
    all_splits = ["train", "validation"] # no test not used
    if all([os.path.exists(f"{data_dir}/{split}.csv") for split in all_splits]):
        df_paths = [f"{data_dir}/{split}.csv" for split in all_splits]
    else:
        raise ValueError(f"Couldn't find train.csv, validation.csv or test.csv in folder in "
                         f"{data_dir}: {os.listdir(data_dir)}")
    save_paths = [f"{output_path}/{model_name}/{split}.csv" for split in all_splits]
    do_device_auto = True
    for exclusion in NO_DEVICE_MAP_MODEL_SUBSTRINGS:
        if exclusion in model_name_or_path:
            do_device_auto = False
            break
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if do_device_auto:
        generator = pipeline(task, model=model_name_or_path, tokenizer=tokenizer, device_map="auto")
    else:
        generator = pipeline(task, model_name_or_path, tokenizer=tokenizer, device=CUDA_DEVICE_INDEX)
    print(f"Starting Generation to {output_path}/{model_name}")
    for df_path, save_path in zip(df_paths, save_paths):
        generate(generator, task, df_path, save_path, max_new_tokens, max_points=max_points, n_gens=n_gens,
                 constraint_max_tokens=constraint_max_tokens)


def generate(generator, task, df_path, save_path, max_new_tokens, max_points=None, n_gens=1, constraint_max_tokens=None):
    df = pd.read_csv(df_path)
    if max_points is not None:
        max_points = min(len(df), max_points)
    prompt_col = "prompt" if "prompt" in df.columns else "text"
    assert prompt_col in df.columns
    df = df[~df[prompt_col].isna()].reset_index(drop=True).loc[:max_points]
    for i in range(n_gens):
        df[f"gen_{i}"] = None
    has_warned = False
    for i in tqdm(range(len(df))):
        prompt = df.loc[i, prompt_col]
        if constraint_max_tokens is not None:
            prompt = " ".join(prompt.split(" ")[:constraint_max_tokens])
            df.loc[i, "prompt"] = prompt  # THIS WILL OVERWRITE IF YOU USE IT ON A TASK DATASET SO WE DON'T ALLOW
        prompt = [prompt] * n_gens
        if task == "text-generation":
            out = generator(prompt, do_sample=True, max_new_tokens=max_new_tokens, return_full_text=False,
                            temperature=1, pad_token_id=generator.tokenizer.eos_token_id)
            out = [o[0]["generated_text"] for o in out]
        elif task == "translation":
            out = generator(prompt, do_sample=True, temperature=1, pad_token_id=generator.tokenizer.eos_token_id)
            out = [o[0]["translation_text"] for o in out]
        else:
            out = generator(prompt, do_sample=True, temperature=1, pad_token_id=generator.tokenizer.eos_token_id)
            key = list(out[0][0].keys())[0]
            out = [o[0][key] for o in out]
            if not has_warned:
                warnings.warn(f"When generating for unspecified task {task}, inferred to use {key}. "
                              f"Manually specify behavior if needed and this could fail")
                has_warned = True
        for j in range(n_gens):
            df.loc[i, f"gen_{j}"] = out[j]
    l1 = len(df)
    df = df.dropna().reset_index(drop=True)
    l2 = len(df)
    print(f"Dropped: {l1-l2} NaNs after generation")
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()

