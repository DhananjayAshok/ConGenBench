import click
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from prompting import get_prompt
from prompt_models import PromptModel
import warnings

@click.command()
@click.option('--data_folder', default=None, type=str, help="path to folder in generated_data with "
                                                            "[train, validation].csv files and gen column")
@click.option("--n_gen_score", default=1, type=int, help="Number of the generations to score")
@click.option('--constraint', default=None, type=None, help="The kind of constraint we care about")
@click.option("--model_name", default=None, type=str, help="LLM to prompt for data")
@click.option("--score", default=False, type=bool, help="If True we predict scores from 1-10 else True or False")
@click.option("--k", default=0, type=int, help="Number of examples to use as FewShot")
@click.option("--cot", default=False, type=bool, help="To elicit reasoning as well as output")
@click.option("--output_only", default=False, type=bool, help="If true will score only the gen and not the prompt")
@click.option("--author", default=None, type=str, help="If we want only prompts written by a specific author")
@click.option('--random_seed', default=42, type=int, help="Random Seed Value")
@click.option("--max_points", default=None, type=int, help="If specified the first max_points from each file are taken")
@click.option('--checkpoint', default=False, type=bool, help="True if you are completing a previous run")
def main(data_folder, n_gen_score, constraint, model_name, score, k, cot, output_only, author, random_seed, max_points, checkpoint):
    np.random.seed(random_seed)
    model = PromptModel(model_name=model_name, score=score, coT=cot)
    dir = data_folder+f'/{constraint}/{model_name}_score_{score}_k_{k}_coT_{cot}_author_{author}_max_points_{max_points}_seed_{random_seed}'
    if not checkpoint:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"Script halted because folder {dir} already exists. "
                  f"Run with the --checkpoint True flag if you want to complete a previous run, "
                  f"Delete it if you dont care about losing data or rename it to preserve both versions")
            exit()
    else:
        if not os.path.exists(dir):
            raise ValueError(f"--checkpoint flag included but {dir} does not exist")
    print(f"Starting Generation to directory: {dir}")
    splits = ["train", "validation"]
    checkpoint_paths = [os.path.join(dir, f"{split}.csv") for split in splits]
    df_paths = [os.path.join(data_folder, f"{split}.csv") for split in splits]
    assert all([os.path.exists(df_path) for df_path in df_paths]), f"Could not find one of {df_paths}"
    # by here df_paths is a list that points to a df which must have a gen column
    prompt = get_prompt(constraint, author=author, score=score, coT=cot, k=k, random_seed=random_seed)
    for i in range(len(splits)):
        generate_data(model, prompt, output_only, df_paths[i], checkpoint_paths[i], n_gen_score=n_gen_score, max_points=max_points)


def infer_n_gen_score(df, n_gen_score):
    n_cols = sum([int("gen" in col) for col in df])
    to_ret = min(n_cols, n_gen_score)
    if to_ret < n_gen_score:
        warnings.warn(f"n_gen_score set to {n_gen_score} but only {n_cols} of the columns {df.columns}")
    return to_ret


def generate_data(model, prompt, output_only, df_path, checkpoint_path, n_gen_score=1, checkpoint_every=100, max_points=None):
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)[:max_points]
        n_gen_score = infer_n_gen_score(df, n_gen_score)
    else:
        df = pd.read_csv(df_path)
        n_gen_score = infer_n_gen_score(df, n_gen_score)
        for i in range(n_gen_score):
            df[f"pred_{i}"] = None
        gen_cols = [f"gen_{i}" for i in range(n_gen_score)]
        nanindex = df[df[gen_cols].isna().any(axis=1)].index
        df = (df.drop(nanindex, axis=0)[:max_points]).reset_index(drop=True)
    pred_cols = [f"pred_{i}" for i in range(n_gen_score)]
    gen_cols = [f"gen_{i}" for i in range(n_gen_score)]

    start = df[df[pred_cols].isna().any(axis=1)].index.min()  # pred_0 must be defined and Nans will be same
    if np.isnan(start):
        print(f"Restarted checkpoint but found no nans in pred columns, saving df with {len(df)} points...")
        df.to_csv(checkpoint_path, index=False)
    else:
        if start > 0:
            print(f"Restarting at {start}/{len(df)}...")
        for i in tqdm(range(start, len(df))):
            for j in range(n_gen_score):
                if pd.isna(df.loc[i, f"gen_{j}"]):
                    out = None
                else:
                    if output_only:
                        text = df.loc[i, f"gen_{j}"]
                    else:
                        text = df.loc[i, "prompt"] + " " + df.loc[i, f"gen_{j}"]
                    in_prompt = prompt.replace("[text]", text)
                    out = model(in_prompt)
                df.loc[i, f"pred_{j}"] = out
            if i % checkpoint_every == 0 or i == len(df)-1:
                if i == len(df) - 1:
                    curr_len = len(df)
                    nanindex = df[df[gen_cols + pred_cols + ["prompt"]].isna().any(axis=1)].index
                    df = df.drop(nanindex).reset_index(drop=True)

                    new_len = len(df)
                    if new_len - curr_len > 0:
                        print(f"After Prompt Generation there were {new_len-curr_len} NaNs, dropping...")
                df.to_csv(checkpoint_path, index=False)
    return


if __name__ == "__main__":
    main()
