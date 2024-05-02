import click
import os
import pandas as pd
import numpy as np


@click.command()
@click.option('--data_dir', type=str, help="path to folder in generated_data with "
                                                            "[train, validation].csv files , gen_k columns and pred_k columns")
@click.option('--include_prompt', default=True, type=bool, help="If false will not include prompt in text column")
@click.option('--fudge', default=False, type=bool, help="If True will also unroll all partial sequences with the same label")
@click.option('--condgen', default=False, type=bool, help="If True will unroll into a prompt, gen and label column")
@click.option("--max_points", default=None, type=int, help="If specified the first max_points from each file are taken")
@click.option('--random_seed', default=42, type=int, help="Random Seed Value")
def unravel(data_dir, include_prompt, fudge, condgen, max_points, random_seed):
    np.random.seed(random_seed)
    new_data_dir = data_dir + f"_unravel"
    os.makedirs(new_data_dir, exist_ok=True)
    for split in ["train", "validation"]:
        df_path = os.path.join(data_dir, f"{split}.csv")
        new_df_path = os.path.join(new_data_dir, f"{split}.csv")
        df = pd.read_csv(df_path)
        n_gens = 0
        n_preds = 0
        for col in df.columns:
            if "gen" in col:
                n_gens += 1
            if "pred" in col:
                n_preds += 1
        n_gens = min(n_gens, n_preds)
        if not condgen:
            new_df = pd.DataFrame(columns=["text", "label"], data=[])
        else:
            new_df = pd.DataFrame(columns=["prompt", "gen", "label"], data=[])
        for i in range(n_gens):
            df["label"] = df[f"pred_{i}"]
            if condgen:
                df["gen"] = df[f"gen_{i}"]
            if include_prompt:
                df["text"] = df["prompt"] + " " + df[f"gen_{i}"]
            else:
                df["text"] = df[f"gen_{i}"]
            if condgen:
                new_df = pd.concat((new_df, df[["prompt", "gen", "label"]]), ignore_index=True)
            else:                           
                new_df = pd.concat((new_df, df[["text", "label"]]), ignore_index=True)
        if max_points is not None and len(new_df) > max_points:
            new_df = new_df.sample(max_points).reset_index(drop=True)
        if fudge:
            assert not condgen
            data = []
            for i in range(len(new_df)):
                text = new_df.loc[i, "text"]
                label = new_df.loc[i, "label"]
                splitted = text.split(" ")
                for j, item in enumerate(splitted):
                    candidate = " ".join(splitted[:j])
                    if len(candidate.strip()) > 0:
                        data.append([candidate, label])
            new_df = pd.DataFrame(columns=["text", "label"], data=data)
            new_df.to_csv(new_df_path, index=False)
            new_df = pd.read_csv(new_df_path)
            new_df = new_df.dropna(axis=0)
        new_df['label'] = new_df['label'].astype(int)
        new_df.to_csv(new_df_path, index=False)


if __name__ == "__main__":
    unravel()
