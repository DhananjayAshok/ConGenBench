from tqdm import tqdm
import pandas as pd
import numpy as np

prefix = f"[INST]Given the following text |'[text]'|"
score_instr = f" answer with a score from 1-10"
bool_instr = f" answer with either 'False' or 'True'"
reasoning_instr = " provide your reasoning."
constraints = {"sentiment": " does the text have a positive sentiment?",
              "toxicity": " is the text toxic?",
              "genre": " is the genre of the text horror (not  romance)?",
              "formality": " is the text using formal language?",
               "factuality": " is the text factually accurate?",
              "clickbait": " is the text in a clickbait style?",
               "excitement": " is the text exciting as opposed to boring?",
               "sensationalism": " is the text sensationalistic as opposed to reserved?",
               "humor": " is the text funny or humorous?",
               "paggressive": "is the text passive aggressive?",
               "satire": "is the text satirical?",
               "irony": "is the text ironic?",
               "topic": "is the topic of the text World Events?"
               }


def get_zs_prompt(constraint, score=False, coT=False):
    assert constraint in constraints
    instr_str = score_instr if score else bool_instr
    cot_str = reasoning_instr if coT else ""
    return prefix + instr_str + constraints[constraint] + cot_str + "[/INST]\nAnswer: "


def get_fs_prompt(constraint, texts, targets, explanations=None, score=None):
    assert len(texts) == len(targets)
    if explanations is not None:
        assert len(explanations) == len(texts)
        coT = True
    else:
        coT = False
    if score is not None:
        if len(texts) == 0:
            return get_zs_prompt(constraint, score=score, coT=coT)
    else:
        score = isinstance(targets[0], int)
    prompt = ""
    zs = get_zs_prompt(constraint, score=score, coT=coT)
    for i in range(len(texts)):
        text = texts[i]
        target = targets[i]
        prompt = prompt + zs.replace("[text]", text)
        if coT:
            prompt = prompt + f"{explanations[i]} | {target}\n"
        else:
            prompt = prompt + f"{target}\n"
    prompt = prompt + zs
    return prompt


def get_demonstrations(constraint, author=None):
    try:
        df = pd.read_csv("prompt_demonstrations.csv")
    except FileNotFoundError:
        df = pd.read_csv("../prompt_generation/prompt_demonstrations.csv")
    assert constraint in constraints
    df = df[df["constraint"] == constraint]
    if author is not None and author.lower() != "none":
        df = df[df["author"] == author]
    if len(df) == 0:
        raise ValueError(f"No Demonstrations found in csv for constraint: {constraint}" +
                         f"with author {author}" if author is not None else "")
    return df


def get_prompt(constraint, author=None, score=False, coT=False, k=3, random_seed=42):
    if k == 0:
        return get_zs_prompt(constraint, score=score, coT=coT)
    np.random.seed(random_seed)
    df = get_demonstrations(constraint, author=author)
    if k > len(df):
        raise ValueError(f"Tried running k={k} with only {len(df)} demonstrations available")
    if not score:
        df["score"] = (df["score"].astype(int) > 6).astype(str)  # 6 is an annoying hyperparameter here but I want to give the data some bias towards identifying positives. Have not tuned this. 
    texts = df["text"].tolist()
    targets = df["score"].tolist()
    if not coT:
        explanations = None
    else:
        explanations = df["explanation"].tolist()
    return get_fs_prompt(constraint, texts, targets, explanations=explanations, score=score)
