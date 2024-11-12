import pandas as pd
import numpy as np


# the two sets have romance, horror and fantasy in common, for now am picking two for binary classification

def handle_tagmybook():
    df = pd.read_csv(f"genre/tagmybook.csv")
    df["text"] = df["synopsis"]
    df = df[df["genre"].isin([genre1, genre2])].reset_index(drop=True)
    df["label"] = df["genre"].map({genre1: 0, genre2: 1})
    df = df[["text", "label"]]
    df.to_csv(f"genre/tagmybook.csv", index=False)


def handle_storycontrol():
    dfs = [pd.read_csv(file, delimiter="\t") for file in ["cls_train.tsv", "cls_dev.tsv"]]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["text"] = df["content"]
    df = df[df["genre"].isin([genre1, genre2])].reset_index(drop=True)
    df["label"] = df["genre"].map({genre1: 0, genre2: 1})
    df = df[["text", "label"]]
    df.to_csv(f"genre/storycontrol.csv", index=False)


if __name__ == "__main__":
    genre1 = "romance"
    genre2 = "horror"
    handle_tagmybook()
    handle_storycontrol()
