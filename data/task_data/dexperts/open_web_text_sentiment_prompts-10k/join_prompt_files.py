import pandas as pd
import numpy as np
np.random.seed(42)
for split in ["train", "validation", "test"]:
    df = pd.concat([pd.read_csv(f"{sent}_prompts/{split}.csv") for sent in ["positive", "negative", "neutral"]])
    df = df.sample(len(df)).reset_index(drop=True)
    df.to_csv(f"{split}.csv", index=False)
