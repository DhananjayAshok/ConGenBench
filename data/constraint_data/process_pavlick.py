import pandas as pd
from datasets import load_dataset, disable_caching

if __name__ == "__main__":
    disable_caching()
    dataset = load_dataset("osyvokon/pavlick-formality-scores")
    df = pd.concat([dataset[split].to_pandas() for split in dataset], axis=0, ignore_index=True)
    df = df[(df.avg_score <= -1) | (df.avg_score >= 1)]
    df["text"] = df["sentence"]
    df["label"] = (df.avg_score > 0).astype(int)
    df = df[["text", "label"]]
    df.to_csv(f"formality/pavlick.csv", index=False)
