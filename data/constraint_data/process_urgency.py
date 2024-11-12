import pandas as pd
import os

if __name__ == "__main__":
    assert_str = "You must download the urgency_labels.tsv file from: " \
                 "https://github.com/niless/urgency or https://app.box.com/s/vbk04ujt2jw9z01vssxxhxozpnbeb61k and " \
                 "place it under constraint_data"
    assert "urgency_labels.tsv" in os.listdir(), assert_str
    filepaths = []
    for file in os.listdir():
        if ".csv" in file:
            filepaths.append(file)
    dfs = [pd.read_csv(f"{file}")[["choose_one_category", "tweet_id", "tweet_text"]] for file in filepaths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df[~df["tweet_id"].isna()].reset_index(drop=True)
    df["tweet_id"] = df["tweet_id"].apply(lambda x: x.strip("'")).astype(int)
    urgency_df = pd.read_csv("urgency_labels.tsv", delimiter="\t")
    joined_df = pd.merge(urgency_df, df, on="tweet_id")
    joined_df.rename({"tweet_text": "text", "urgency": "label"}, axis=1, inplace=True)
    joined_df.drop("tweet_id", axis=1, inplace=True)
    joined_df = joined_df[joined_df["label"].isin(["not_urgent", "definitely_urgent", "extremely_urgent"])].reset_index(drop=True)
    not_urgent = joined_df["label"] == "not_urgent"
    joined_df["label"] = 1
    joined_df.loc[not_urgent, "label"] = 0
    joined_df["label"] = joined_df["label"].astype(int)
    joined_df = joined_df[["text", "label"]]
    joined_df.to_csv(f"urgency/urgency.csv", index=False)


