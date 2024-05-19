###########################################################################
"""
This script will split the datasets of both task and constraint data and save it as [train, validation, test].csv files
For all tasks, the text that is supposed to be used as input to the generative model is in the "prompt" column.
If there is an associated "true" output or set of references they are stored in the "target" column
In general you should not assume that these are the only two columns in the DataFrame/CSV.

To add your own dataset to this you must do the following:
1A.  Add the dataset to either the TASK or CONSTRAINT dataset list below if it is a huggingface dataset.
    This is only the dataset name not the provider so "SetFit/sst5" is just "sst5" for example
1B. If the dataset is not from huggingface give the string that can uniquely identify this dataset in TASK_FILE_MARKERS
    If the dataset is a constraint dataset you can skip this
2.  If the dataset has its own train, test and validation splits then add it to the NO_SPLIT list.
    This is not required if the other steps are done properly but is a good safety measure
3.  For a constraint dataset you must add the constraint type to the CONSTRAINT_TYPES dictionary
4.  Then go into the get_huggingface_(task/constraint)_dataset function and handle your dataset like the others
    See how the returned values are dealt with in the main function. Make sure to use standard column names.
"""

###########################################################################
import pandas as pd
import numpy as np
import warnings
from datasets import load_dataset, disable_caching
import click
import os
import re

CONSTRAINT_DATASETS = ["imdb", "yelp_polarity", "sst2", "sst5", "cola", "sms_spam", "spamassassin"]
TASK_DATASETS = ["real-toxicity-prompts", "common_gen", "bookcorpus", "opus100", "opus_books", "tatoeba",
                 "cnn_dailymail", "gigaword", "xsum", "eli5", "scifact", "fever", "squad"]
TASK_FILE_MARKERS = ["dexperts", "writing-prompts", "factuality-prompts", "roc-stories"]
NO_SPLIT = ["opus100", "cnn_dailymail", "common_gen", "gigaword", "xsum",
            "train.wp_source", "valid.wp_source", "test.wp_source"]
CONSTRAINT_TYPES = {"sentiment": ["imdb", "yelp_polarity", "sst2", "sst5"], "grammar": ["cola"],
                    "spam": ["sms_spam", "spamassassin"]}

############################################################################
# Dataset Specific Hyperparameters
AG_NEWS_CLASS = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}["World"]
WRITING_PROMPTS_PROMPT = "[INST]Given the prompt above, write a 2 sentence story[/INST]"
CNN_GEN_COLUMN = ["article", "highlights"][0]
CNN_GEN_N_WORDS = 15  # CNN Dailymail task will use the first few words (judged by splitting on whitespace) as a prompt
CNN_SUM_N_SENTS = 8
CNN_SUM_PROMPT = "[INST]Write a short summary for the above paragraph[/INST]"
XSUM_GEN_COLUMN = ["document", "summary"][0]
XSUM_GEN_N_WORDS = 15  # XSUM task will use the first few words as a prompt
TRANSLATION_LANG_SOURCE = "es"  # The code will error out if the split isn't found
TRANSLATION_LANG_TARGET = "en"  # Change from english to something else at your own peril
SCIFACT_EVIDENCE_PROMPT = True  # Flag to insert evidence sentences into SciFact prompt
EC_EVIDENCE_MARKER = "Evidence: "
EC_CLAIM_MARKER = "\nClaim: "
EC_PROMPT = "[INST] Modify the incorrect claim so that is in agreement with the evidence above[/INST]"
ELI5_PROMPT = "[INST]Write a short answer to the above question [/INST]"
ROC_STORIES_N_SENT = min(3, 5)  # The number of story sentences that will be in the prompt of ROC stories. 5 is max
SQUAD_MAX_SENT = 10  # The maximum number of sentences allowed in a squad prompt.
SQUAD_PROMPT = "[INST] Given the above paragraph, come up with a question [/INST]"
############################################################################


def get_huggingface_constraint_dataset(dataset_name):
    if dataset_name in ["imdb", "yelp_polarity"]:  # these datasets have a clean
        # train, test set (might have val) with text, label cols
        dataset = load_dataset(dataset_name)
        sets = ["train", "test"]
        for valname in ["val", "validation"]:
            if valname in dataset:
                sets.append(valname)
        dfs = [dataset[set_name].to_pandas() for set_name in sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)
    elif dataset_name == "ag_news":
        dataset = load_dataset(dataset_name)
        df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()])
        corr = df["label"] == AG_NEWS_CLASS
        df.loc[corr, "label"] = 1
        df.loc[~corr, "label"] = 0
        df["label"] = df["label"].astype(int)
    elif dataset_name == "sst2":
        dataset = load_dataset(dataset_name)
        sets = ["train", "validation", "test"]
        dfs = [dataset[set_name].to_pandas() for set_name in sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.drop("idx", axis=1).rename({"sentence": "text"}, axis=1)
    elif dataset_name == "sst5":
        dataset = load_dataset("SetFit/sst5")
        sets = ["train", "validation", "test"]
        dfs = [dataset[set_name].to_pandas() for set_name in sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.drop("label_string", axis=1)  # TODO: BUG HERE
        df = df[df["label"].isin([0, 4])]
        df.dropna(inplace=True)
        df["label"] = df["label"].map({0: 0, 4: 1})
    elif dataset_name == "cola":
        dataset = load_dataset("glue", "cola")
        sets = ["train", "validation"]  # test has no labels
        dfs = [dataset[set_name].to_pandas() for set_name in sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.drop("index", axis=1)  # TODO: BUG HERE
        df["label"] = (df["label"] == 0).astype(int)  # We want non-grammatical to be 1 as target
        df.rename({"sentence": "text"}, axis=1, inplace=True)
    elif dataset_name == "sms_spam":
        dataset = load_dataset(dataset_name)
        df = dataset["train"].to_pandas().rename({"sms": "text"}, axis=1)
    elif dataset_name == "spamassassin":
        dataset = load_dataset("talby/spamassassin")
        df = dataset["train"].to_pandas().drop("group", axis=1)
    else:
        raise ValueError(f"Unrecognized dataset: {dataset_name}")
    return df


def get_huggingface_task_dataset(dataset_name):
    filenames_single = [dataset_name]
    filenames_split = [f"{dataset_name}/{split}" for split in ["train", "validation", "test"]]
    if dataset_name == "real-toxicity-prompts":
        df = load_dataset("allenai/real-toxicity-prompts")["train"].to_pandas()
        non_challenge = df[~df["challenging"]].reset_index(drop=True)
        df = df[df["challenging"]].reset_index(drop=True)  # comment these lines out to just use the dataset
        non_challenge = non_challenge.loc[:10*len(df)]
        df = pd.concat([df, non_challenge], axis=0, ignore_index=True)
        df["prompt"] = df["prompt"].apply(lambda x: x["text"])
        df["continuation"] = df["continuation"].apply(lambda x: x["text"])
        df = df[["prompt", "continuation"]]
        dfs = [df]
    elif dataset_name == "bookcorpus":
        df = load_dataset(dataset_name)["train"].to_pandas()
        df.rename({"text": "prompt"}, axis=1, inplace=True)
        dfs = [df]
    elif dataset_name == "common_gen":
        dataset = load_dataset(dataset_name)
        train_df = dataset["train"].to_pandas().drop("concept_set_idx", axis=1)
        valid_df = dataset["validation"].to_pandas().drop("concept_set_idx", axis=1)
        test_df = dataset["test"].to_pandas().drop("concept_set_idx", axis=1)
        dfs = [train_df, valid_df, test_df]
        for df in dfs:
            df["prompt"] = df["concepts"].apply(lambda x: str(x) + " ")
    elif dataset_name == "opus100":
        dataset = load_dataset("opus100", f"{TRANSLATION_LANG_TARGET}-{TRANSLATION_LANG_SOURCE}")
        splits = ["train", "validation", "test"]
        dfs = []
        for df in [dataset[split].to_pandas() for split in splits]:
            df["prompt"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_SOURCE}"])
            df["target"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_TARGET}"])
            df.drop("translation", axis=1, inplace=True)
            dfs.append(df)
    elif dataset_name == "opus_books":
        dataset = load_dataset("opus_books", f"{TRANSLATION_LANG_TARGET}-{TRANSLATION_LANG_SOURCE}")
        splits = ["train"]
        dfs = []
        for df in [dataset[split].to_pandas() for split in splits]:
            df["prompt"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_SOURCE}"])
            df["target"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_TARGET}"])
            df.drop(["translation", "id"], axis=1, inplace=True)
            dfs.append(df)
    elif dataset_name == "tatoeba":
        dataset = load_dataset("tatoeba", lang1=f"{TRANSLATION_LANG_TARGET}", lang2=f"{TRANSLATION_LANG_SOURCE}")
        df = dataset["train"].to_pandas()
        df["prompt"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_SOURCE}"])
        df["target"] = df["translation"].apply(lambda x: x[f"{TRANSLATION_LANG_TARGET}"])
        df.drop(["translation", "id"], axis=1, inplace=True)
        dfs = [df]
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset(dataset_name, '3.0.0')
        splits = ["train", "validation", "test"]
        dfs = []
        sum_dfs = []

        def clean_cnn(x):
            stripped = x.strip()
            sentences = stripped.split(".")
            to_start = 0
            if sentences[0] == "By ":
                to_start += 2
            flag = True
            while flag:
                options = ["Last updated", "EST", "PUBLISHED", "UPDATED", "|"]
                flag = False
                for opt in options:
                    if opt in sentences[to_start]:
                        to_start += 1
                        flag = True
                        break
            return ".".join(sentences[to_start:])
        for df in [dataset[split].to_pandas() for split in splits]:
            df["article"] = df["article"].apply(clean_cnn)
            dfi = pd.DataFrame(columns=["prompt"])
            dfi["prompt"] = df[CNN_GEN_COLUMN].apply(lambda x: " ".join(x.split(" ")[:CNN_GEN_N_WORDS]))
            dfs.append(dfi)
            df["article"] = df["article"].apply(lambda x: ".".join(x.split(".")[:CNN_SUM_N_SENTS]))+CNN_SUM_PROMPT
            df = df.rename({"article": "prompt", "highlights": "target"}, axis=1).drop(["id"], axis=1)
            sum_dfs.append(df)
        dfs.extend(sum_dfs)
        filenames_split.extend([f"{dataset_name}_summ/{split}" for split in ["train", "validation", "test"]])
    elif dataset_name == "xsum":
        dataset = load_dataset(dataset_name)
        splits = ["train", "validation", "test"]
        dfs = []
        sum_dfs = []
        for df in [dataset[split].to_pandas() for split in splits]:
            dfi = pd.DataFrame(columns=["prompt"])
            dfi["prompt"] = df[XSUM_GEN_COLUMN].apply(lambda x: " ".join(x.split(" ")[:XSUM_GEN_N_WORDS]))
            dfs.append(dfi)
            df = df.rename({"document": "prompt", "summary": "target"}, axis=1).drop(["id"], axis=1)
            sum_dfs.append(df)
        dfs.extend(sum_dfs)
        filenames_split.extend([f"{dataset_name}_summ/{split}" for split in ["train", "validation", "test"]])
    elif dataset_name == "gigaword":
        dataset = load_dataset(dataset_name)
        splits = ["train", "validation", "test"]
        dfs = []
        for df in [dataset[split].to_pandas() for split in splits]:
            df = df.rename({"document": "prompt", "summary": "target"}, axis=1)
            dfs.append(df)
    elif dataset_name == "eli5":
        dataset = load_dataset(dataset_name)
        splits = ["train", "validation", "test"]
        cats = ["eli5", "asks", "askh"]
        dfs = []
        filenames_split = []
        for cat in cats:
            for split in splits:
                df = dataset[f"{split}_{cat}"].to_pandas()
                df = df.rename({"title": "prompt"}, axis=1)
                df["target"] = df["answers"].apply(lambda x: x["text"].tolist())
                df["prompt"] = df["prompt"]+ELI5_PROMPT
                df = df[["prompt", "target"]]
                dfs.append(df)
                filenames_split.append(f"{dataset_name}_{cat}/{split}")
    elif dataset_name == "scifact":
        claims = load_dataset("allenai/scifact", 'claims')
        dfs = []
        if SCIFACT_EVIDENCE_PROMPT:
            def get_sent(abstract, ev_sent):
                return " ".join(abstract[ev_sent].tolist()).strip()
            corpus = load_dataset("allenai/scifact", 'corpus')["train"].to_pandas()
            corpus.rename({'doc_id': 'evidence_doc_id'}, axis=1, inplace=True)
            corpus["evidence_doc_id"] = corpus["evidence_doc_id"].astype(int)
            claim_df = pd.concat([claims[split].to_pandas() for split in ["train", "validation"]],
                                 axis=0, ignore_index=True)
            claim_df = claim_df[claim_df["evidence_label"] == "CONTRADICT"].reset_index(drop=True)
            claim_df = claim_df[~claim_df["evidence_sentences"].isna()]
            claim_df = claim_df[claim_df["evidence_doc_id"].apply(lambda x: x.isnumeric())].reset_index(drop=True)
            claim_df['evidence_doc_id'] = claim_df['evidence_doc_id'].astype(int)
            joined = pd.merge(claim_df, corpus, on='evidence_doc_id').reset_index(drop=True)
            joined["evidence"] = joined.apply(lambda x: get_sent(x["abstract"], x["evidence_sentences"]), axis=1)
            joined["prompt"] = f"{EC_EVIDENCE_MARKER}" + joined["evidence"] + f"{EC_CLAIM_MARKER}" + joined["claim"]
            df = pd.DataFrame(columns=["prompt"])
            df["prompt"] = joined["prompt"].unique()
            dfs.append(df)
        else:
            claim_df = pd.concat([claims[split].to_pandas() for split in ["train", "validation"]],
                                 axis=0, ignore_index=True)
            claim_df = claim_df[claim_df["evidence_label"] == "CONTRADICT"].reset_index(drop=True)
            claim_df["prompt"] = f"{EC_CLAIM_MARKER}" + claim_df["claim"]
            df = pd.DataFrame(columns=["prompt"])
            df["prompt"] = claim_df["prompt"].unique()
            dfs.append(df)
    elif dataset_name == "fever":
        dataset = load_dataset("fever", "v1.0")
        wiki = load_dataset("fever", "wiki_pages")['wikipedia_pages'].to_pandas()
        wiki.rename({"id": "evidence_wiki_url", "text": "evidence"}, axis=1, inplace=True)
        splits = ["train", "paper_dev", "paper_test"]
        dfs = []
        for split in splits:
            claim_df = dataset[split].to_pandas()
            claim_df = claim_df[claim_df["label"] == "REFUTES"].reset_index(drop=True)
            claim_df = pd.merge(claim_df, wiki, on='evidence_wiki_url').reset_index(drop=True)
            claim_df = claim_df[~claim_df["evidence"].isna()]
            claim_df = claim_df[~claim_df["claim"].isna()].reset_index(drop=True)
            claim_df["prompt"] = f"{EC_EVIDENCE_MARKER}" + claim_df["evidence"] + f"{EC_CLAIM_MARKER}" + claim_df["claim"] + f"{EC_PROMPT}"
            df = pd.DataFrame(columns=["prompt"])
            df["prompt"] = claim_df["prompt"].unique()
            dfs.append(df)
    elif dataset_name == "squad":
        dfs = []
        dataset = load_dataset("squad_v2")
        train_df = dataset["train"].to_pandas()
        val_df = dataset["validation"].to_pandas()
        joined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        df = pd.DataFrame(columns=["prompt", "target"])
        df["prompt"] = joined_df.groupby("context")["context"].agg(list).apply(lambda x: list(x)[0]).reset_index(drop=True)
        df["target"] = joined_df.groupby("context")["question"].agg(list).reset_index(drop=True)
        droprows = df['prompt'][df["prompt"].apply(lambda x: len(x.split("."))) > SQUAD_MAX_SENT].index
        df = df.drop(droprows, axis=0).reset_index(drop=True)
        df["prompt"] = df["prompt"]+SQUAD_PROMPT
        dfs.append(df)
    else:
        raise ValueError(f"Unrecognized dataset: {dataset_name}")
    if len(dfs) == 1:
        filenames = filenames_single
    elif len(dfs) % 3 == 0:
        filenames = filenames_split
    else:
        raise ValueError(f"Dataset: {dataset_name} has a weird number of splits: {len(dfs)}")
    # assert len(dfs) == len(filenames)
    return dfs, filenames


def get_local_task_dataset(filepath, filename):
    if "open_web_text_sentiment_prompts-10k" in filepath:
        filenames = [f"dexperts/open_web_text_sentiment_prompts-10k/{filename}"]
        df = pd.read_json(filepath, lines=True)
        df["prompt"] = df["prompt"].apply(lambda x: x["text"])
        df["continuation"] = df["continuation"].apply(lambda x: x["text"])
        df.drop(["md5_hash", "num_positive"], axis=1, inplace=True)
        dfs = [df]
    elif "writing-prompts" in filepath:
        columns = ["prompt"]
        data = []
        with open(filepath) as f:
            for line in f:
                data.append([re.sub("\[.*?\]", "", line.strip()).strip()])
        df = pd.DataFrame(data=data, columns=columns)
        df["prompt"] = df["prompt"] + WRITING_PROMPTS_PROMPT
        dfs = [df]
        if filename == "valid":
            filename = "validation"
        filenames = [f"writing-prompts/{filename}"]
    elif "factuality-prompts" in filepath:
        df = pd.read_json(filepath, lines=True)
        df = df[["prompt"]]
        dfs = [df]
        filenames = [f"factuality-prompts/{filename}"]
    elif filename == "jigsaw_nontoxic_prompts-10k":
        df = pd.read_json(filepath, lines=True)
        df["prompt"] = df["prompt"].apply(lambda x: x["text"])
        df["continuation"] = df["continuation"].apply(lambda x: x["text"])
        df = df[["prompt", "continuation"]]
        dfs = [df]
        filenames = [f"dexperts/{filename}"]
    elif filename == "roc":
        df = pd.read_csv(filepath)
        def roc_prompt(rows):
            sentence = ""
            for i in range(ROC_STORIES_N_SENT):
                sentence = sentence + " " + rows[f"sentence{i+1}"]
            return sentence
        df["prompt"] = df.apply(roc_prompt, axis=1)
        dfs = [df[["prompt"]]]
        filenames = [f"roc-stories/"]
    else:
        raise ValueError(f"Unrecognized dataset: {filename} under filepath: {filepath}")
    return dfs, filenames


def load_constraint_data(filepath, dataset_name, binarize_labels, binarize_threshold):
    if filepath is not None:
        assert filepath[-4:] == ".csv", "Are you sure you put in a csv filepath?"
        data_path, file = os.path.split(filepath)
        constraint_type = os.path.split(data_path)[1]
        df = pd.read_csv(filepath)
        filename = constraint_type + "/"+file[:-4]
    else:  # then dataset name is not None
        # constraint_dict = {""}
        dataset_name = dataset_name.split("/")[-1]
        constraint_type = None
        for constraint_type_option in CONSTRAINT_TYPES:
            if dataset_name in CONSTRAINT_TYPES[constraint_type_option]:
                constraint_type = constraint_type_option
                break
        if constraint_type is None:
            raise ValueError(f"Unrecognized Constraint type for dataset: {dataset_name}. Must be added to the CONSTRAINT_TYPES dictionary: {CONSTRAINT_TYPES}")
        df = get_huggingface_constraint_dataset(dataset_name)
        filename = f"{constraint_type}/" + dataset_name
    # By here df is the mixed train, val and test split of any dataset we care about with 2 columns - text, label
    if binarize_labels:
        beyond = df["label"] > binarize_threshold
        df.loc[beyond, "label"] = 1
        df.loc[~beyond, "label"] = 0
        df["label"] = df["label"].astype(int)
    save_path = os.path.join("constraint_data", filename)
    return [df], [save_path]


def load_task_data(filepath, dataset_name):
    if filepath is not None:
        data_path, file = os.path.split(filepath)
        filename = ".".join(file.split(".")[:-1])
        dfs, filenames = get_local_task_dataset(filepath, filename)
    else:  # then dataset name is not None
        dfs, filenames = get_huggingface_task_dataset(dataset_name)
    save_paths = [os.path.join("task_data", filename) for filename in filenames]
    return dfs, save_paths


@click.command()
@click.option('--filepath', default=None, type=str, help="path to csv file either absolute or relative from the "
                                                         "constraint_data/task_data dirs "
                                                         "(based on the --constraint flag)")
@click.option('--dataset_name', default=None,
              type=click.Choice(CONSTRAINT_DATASETS+TASK_DATASETS,
                                case_sensitive=False),
              help="name of a huggingface dataset")
@click.option("--binarize_labels", default=True, type=bool, help="If set to True any continuous labels in the "
                                                                  "data will be discretized to {0, 1}. "
                                                                  "Expects a label column")
@click.option("--binarize_threshold", default=0, type=float, help="If labels are binarized, any values above this "
                                                                    "threshold is set to 1, rest to 0")
@click.option('--max_points_full', default=None, type=int, help="len(train) + len(val) + len(test) set to "
                                                                "min(len(dataset), max_points_full) ")
@click.option("--do_split", default=True, type=bool, help="If false will keep all in one csv")
@click.option('--train_size', default=0.7, type=float, help="Fraction of training size")
@click.option('--val_size', default=0.15, type=float, help="Fraction of val size")
@click.option('--test_size', default=0.15, type=float, help="Fraction of test size")
@click.option('--random_seed', default=42, type=int, help="Random Seed Value")
def main(filepath, dataset_name,
         binarize_labels, binarize_threshold,
         max_points_full, do_split, train_size, val_size, test_size,
         random_seed):
    assert filepath is not None or dataset_name is not None and not (filepath is not None and dataset_name is not None)
    # Handle the no splitting cases
    if dataset_name is not None:
        if dataset_name in NO_SPLIT:
            if do_split:
                print(f"dataset: {dataset_name} is not recommended with do_split=True, setting to False...")
            do_split = False
    if filepath is not None:
        for no_split in NO_SPLIT:
            if no_split in filepath:
                if do_split:
                    print(f"dataset: {dataset_name} is not recommended with do_split=True, setting to False...")
                do_split = False
                break

    # Decides whether we are working with constraint or task data
    constraint = None
    if dataset_name is not None:
        if dataset_name in CONSTRAINT_DATASETS:
            constraint = True
        elif dataset_name in TASK_DATASETS:
            constraint = False
        else:
            raise ValueError(f"Unrecognized Dataset: {dataset_name}")
    elif filepath is not None:
        if any([filemarker in filepath for filemarker in TASK_FILE_MARKERS]):
            constraint = False
        else:
            constraint = True  # Assuming this as default if cannot find in task datasets
    else:
        raise ValueError("No dataset_name or filepath, wat to do now?")
    np.random.seed(random_seed)
    disable_caching()
    if constraint:
        dfs, save_paths = load_constraint_data(filepath, dataset_name, binarize_labels, binarize_threshold)
    else:
        dfs, save_paths = load_task_data(filepath, dataset_name)
    if len(dfs) > 1 and do_split:
        warnings.warn(f"For dataset: {dataset_name} got {len(dfs)} dfs with splitting enabled. "
                      f"This is odd and might be a bug")
    for df, save_path in zip(dfs, save_paths):
        df = df.dropna()
        if max_points_full is None:
            n_points = len(df)
        else:
            n_points = min(len(df), max_points_full)
        df = df.sample(n_points).reset_index(drop=True)
        n = len(df)
        flag = False
        for splitname in ["train", "validation", "test"]:
            if splitname == save_path[-len(splitname):]:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                flag = True
                break
        if not flag:
            os.makedirs(save_path, exist_ok=True)
        if not do_split:
            df.to_csv(f"{save_path}.csv", index=False)   # This will save it to train, validation or test.csv
        else:
            tot = train_size + val_size + test_size
            assert tot > 0, "check your split sizes bro"
            train_size = int((train_size/tot) * n)
            val_size = int((val_size/tot) * n)
            test_size = int((test_size/tot) * n)
            if train_size > 0:
                train_df = df.loc[:train_size, :]
                train_df.to_csv(f"{save_path}/train.csv", index=False)
                del train_df
            if val_size > 0:
                val_df = df.loc[train_size:train_size+val_size, :]
                val_df.to_csv(f"{save_path}/validation.csv", index=False)
                del val_df
            if test_size > 0:
                test_df = df.loc[-test_size:, :]
                test_df.to_csv(f"{save_path}/test.csv", index=False)
        print(f"Files written to {save_path}")


if __name__ == "__main__":
    main()

