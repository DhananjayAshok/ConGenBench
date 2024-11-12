import pandas as pd
import numpy as np
import click


@click.command()
@click.option('--attribute', default="toxic", help="Attribute to use as label",
              type=click.Choice(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
                                case_sensitive=False))
@click.option("--preprocess", default=False, type=bool)
def main(attribute, preprocess):
    if preprocess:
        handle_challenge_frame()
    # Method borrowed from:
    # https://github.com/alisawuffles/DExperts/blob/main/scripts/data/create_jigsaw_toxicity_data.py
    unintended_bias_dict = {"toxic": "toxicity", "severe_toxic":"severe_toxicity",
                            "obscene": "obscene", "threat": "threat",
                            "insult": "insult", "identity_hate": "identity_attack"}
    unintended_attribute = unintended_bias_dict[attribute]
    df = pd.read_csv("toxicity/jigsaw-unintended-bias-in-toxicity-classification.csv")
    eq_0 = df[unintended_attribute] == 0
    geq0_5 = df[unintended_attribute] >= 0.5
    df = df[eq_0 | geq0_5].reset_index(drop=True)
    df["text"] = df["comment_text"]
    df["label"] = df[unintended_attribute]
    df = df[["text", "label"]]
    df.to_csv("toxicity/jigsaw_unintended_processed.csv", index=False)

    df = pd.read_csv("toxicity/jigsaw-toxic-comment-classification-challenge.csv")
    df["text"] = df["comment_text"]
    df["label"] = df[attribute].astype(int)
    df = df[["text", "label"]]
    df.to_csv("toxicity/jigsaw_original_processed.csv", index=False)
    return


def handle_challenge_frame():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    test_labels = pd.read_csv("test_labels.csv")
    test_labels.drop("id", axis=1, inplace=True)
    test = pd.concat([test, test_labels], axis=1)
    test.replace(-1, np.nan, inplace=True)
    test = test.dropna(axis=0).reset_index(drop=True)
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df.to_csv("toxicity/jigsaw-toxic-comment-classification-challenge.csv", index=False)


if __name__ == "__main__":
    main()

