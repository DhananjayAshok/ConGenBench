import evaluate
import pandas as pd
import click
import os
from transformers import pipeline, set_seed
from tqdm import tqdm
from nltk import ngrams
from nltk.tokenize import word_tokenize
from googleapiclient import discovery
import json
import time
import warnings
perspective_api_key = os.getenv("PERSPECTIVE_API_KEY")
perspective_quota_per_minute = 60


@click.command()
@click.option('--gen_file', type=str, help="Path to csv with gen columns")
@click.option('--metric', type=click.Choice(["perspective_toxicity", "bleu", "diversity"], case_sensitive=False),
              help="key name of metric")
@click.option('--out_file', default=None, type=str, help="Path to csv with gen columns. "
                                                         "If None writes to genfile but not recommended for cleanliness")
@click.option("--max_points", default=None, type=int, help="If specified the first max_points from each file are taken")
@click.option("--prompt", default=True, type=bool, help="If true the prompt + gen is scored else only gen")
@click.option("--reference_col", default="target", type=str, help="Column of csv to get reference sentence from for metrics")
@click.option("--overwrite", default=False, type=bool, help="Should we rewrite metrics if already in dataframe")
def main(gen_file, metric, out_file, max_points, prompt, reference_col, overwrite):
    if metric in ["bleu"]:
        assert reference_col is not None, f"Reference column cannot be None with metric {metric}"
    if out_file is None:
        out_file = gen_file
    df = pd.read_csv(gen_file)
    if max_points is not None:
        max_points = min(len(df), max_points)
    else:
        max_points = len(df)
    l1 = len(df)
    gen_cols = []
    for column in df.columns:
        if "gen_" in column:
            gen_cols.append(column)
            df = df[~df[column].isna()].reset_index(drop=True)
    l2 = len(df)
    print(f"Dropped {l1-l2} NaNs")
    df = df.loc[:max_points]
    if metric not in ["diversity"]:
        for gen_col in gen_cols:
            do_scoring(df, prompt, metric, gen_col=gen_col, overwrite=overwrite)
    if metric == "diversity":
        assert len(gen_cols) > 1, "Cant assess diversity without multiple generations"
        for i in range(1, 5):
            df[f"distinct_{i}_gram"] = df[gen_cols].apply(lambda x: distinct_n_sentences([x[col] for col in gen_cols],
                                                                                         i), axis=1)
    df.to_csv(out_file, index=False)


def do_scoring(df, prompt, metric, gen_col="gen_0", overwrite=False):
    gen_call_index = gen_col.split("_")[1]
    if prompt:
        to_score = df["prompt"] + " " + df[gen_col]
    else:
        to_score = df[gen_col]
    if metric == "perspective_toxicity":
        write_col = f"perspective_toxicity_{gen_call_index}"
        if not overwrite and write_col in df.columns:
            warnings.warn(f"For metric {metric} on gen_col {gen_col}, column {write_col} already exists. "
                          f"Call with overwrite if you want....")
            return
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=perspective_api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        n_reqs = 0
        for i in tqdm(range(len(df))):
            if n_reqs >= perspective_quota_per_minute:
                time.sleep(60)
                n_reqs = 0
            df.loc[i, write_col] = toxicity(client, to_score[i])
            n_reqs += 1
    elif metric == "bleu":
        write_col = f"bleu_{gen_call_index}"
        if not overwrite and write_col in df.columns:
            warnings.warn(f"For metric {metric} on gen_col {gen_col}, column {write_col} already exists. "
                          f"Call with overwrite if you want....")
            return
        bleu = evaluate.load("bleu")
        df["bleu"] = None
        for i in tqdm(range(len(df))):
            df.loc[i, f"bleu_{gen_call_index}"] = bleu_score(bleu, to_score[i], df.loc[i, reference_col])  # if this fails likely reference column wrong
    else:
        raise ValueError(f'Unrecognized metric {metric}')


def toxicity(client, generation):
    analyze_request = {
        'comment': {'text': f'{generation}'},
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']


def bleu_score(bleu, prediction, reference):
    if isinstance(reference, list):
        pass
    elif reference[0] == "[":  # then we assume its a list saved as object
        reference = eval(reference)
    else:
        reference = [reference]
    return bleu.compute(predictions=[prediction], references=reference)['bleu']


# Shamelessly taken from https://github.com/neural-dialogue-metrics/Distinct-N/blob/main/distinct_n/metrics.py
def distinct_n_sentences(sentences, n):
    """
    Compute distinct-N across sentences.
    :param sentences: a list of sentences.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentences) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set()
    for sent in sentences:
        distinct_ngrams = distinct_ngrams.union(set(ngrams(word_tokenize(sent), n)))
    return len(distinct_ngrams) / len(sentences)


if __name__ == "__main__":
    main()

