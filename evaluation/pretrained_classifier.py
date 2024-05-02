import pandas as pd
import click
import os
from transformers import pipeline, set_seed
from tqdm import tqdm
import warnings


CUDA_DEVICE = 0

@click.command()
@click.option('--model_name_or_path', type=str, help="Name of the model to use for classification. "
                                                                   "Should be able to accept the 'gen' column of "
                                                                   "the dataset with no changes")
@click.option('--gen_file', type=str, help="Path to csv with a gen column")
@click.option('--out_file', type=str, help="Path to csv with a gen column")
@click.option('--score_col_name', default=None, type=str, help="name of score column. If not specified model name is used")
@click.option("--max_points", default=None, type=int, help="If specified the first max_points from each file are taken")
@click.option("--prompt", default=True, type=bool, help="If true the prompt + gen is scored else only gen")
@click.option("--label_target", default=None, type=str, help="The class label whose score we should consider")
def main(model_name_or_path, gen_file, out_file, score_col_name, max_points, prompt, label_target):
    if "generated_data" in gen_file and out_file == gen_file and max_points is not None:
        raise ValueError(f"This script will drop NaN columns and delete generated data if run like this. "
                         f"Either set a different out_file or set max_points to None")
    if label_target is None:
        warnings.warn(f"Label Target argument has not been set, defaults to '1'.  "
                      f"It is best to be explicit as code will run without errors even if '1' is not appropriate ")
        label_target = "1"
    if score_col_name is None:
        score_col_name = model_name_or_path.split("/")[-1]
    try:
        classifier = pipeline("text-classification", model=model_name_or_path, device_map="auto")
    except:
        classifier = pipeline("text-classification", model=model_name_or_path, device=CUDA_DEVICE)
    df_path = gen_file
    classify(classifier, df_path, out_file, label_target, score_col_name=score_col_name,
             max_points=max_points, prompt=prompt)


def classify(classifier, df_path, save_path, label_target, score_col_name, max_points=None, prompt=True):
    df = pd.read_csv(df_path)
    n_cols = sum([int("gen_" in col) for col in df])
    if max_points is not None:
        max_points = min(len(df), max_points)
    else:
        max_points = len(df)
    for col in range(n_cols):
        df = df[~df[f"gen_{col}"].isna()].reset_index(drop=True).loc[:max_points]
    for i in tqdm(range(len(df))):
        for j in range(n_cols):
            if prompt:
                gen = df.loc[i, "prompt"] + " " + df.loc[i, f"gen_{j}"]
            else:
                gen = df.loc[i, f"gen_{j}"]
            out = classifier(gen)[0]
            label = out["label"]
            if str(label).lower() == label_target.lower():
                out = out["score"]
            else:
                out = 1 - out["score"]  # Assuming its binary otherwise little screwed
            df.loc[i, f"{score_col_name}_{j}"] = out
    l1 = len(df)
    df = df.dropna().reset_index(drop=True)
    l2 = len(df)
    print(f"Dropped: {l1-l2} NaNs after classification")
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()

