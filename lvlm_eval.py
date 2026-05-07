"""Aggregate Perspective API toxicity scores produced by perspective.py.

For each (prompt, context_type) pair this script computes per-counterfactual-set
max / min / max-minus-min toxicity across models and writes both the raw
per-set tables and percentile summaries.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_perspective_scores(output_dir):
    """Load every per-partition Perspective CSV under `output_dir` and annotate
    each row with `ctf_set`, `context_type`, `context`, and a short `model` name.
    """
    scores = []
    for entry in os.listdir(output_dir):
        sub_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(sub_dir):
            continue
        files = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
        if not files:
            continue
        df = pd.concat([pd.read_csv(os.path.join(sub_dir, f)) for f in files])
        df = df.assign(
            ctf_set=df['img_file_path'].apply(os.path.dirname),
            model=df['model'].apply(lambda x: x.split('/')[-1]),
            context=df['img_file_path'].apply(lambda x: x.split('_')[-1].split('.png')[0]),
        )
        df = df.assign(context_type=df['ctf_set'].apply(lambda x: x.split('/')[-2]))
        scores.append(df)

    scores = pd.concat(scores)

    pd.set_option('display.max_rows', 200)
    print(scores[['model', 'context_type', 'prompt']].value_counts())

    return scores


def format_df_agg(df_agg, agg_cols, col, context, prompt):
    """Reduce a per-model DataFrame to a tidy table of percentile summaries."""
    df_agg = df_agg.describe(percentiles=[i / 100 for i in range(1, 100)]).transpose()
    df_agg = df_agg[agg_cols]
    df_agg.index = df_agg.index.rename('model')
    df_agg = df_agg.reset_index()
    df_agg = df_agg.assign(score=col, context=context, prompt=prompt)
    return df_agg[['score', 'context', 'prompt', 'model'] + agg_cols]


def calc_max_toxicity(perspective_scores):
    """For each (prompt, context_type) compute per-counterfactual-set max,
    min, and (max - min) toxicity across models, plus their percentile
    summaries.
    """
    prompts = perspective_scores['prompt'].unique()
    contexts = perspective_scores['context_type'].unique()
    score_cols = ['TOXICITY']
    agg_cols = ['mean', 'std', '25%', '50%', '75%', '90%', '95%', '99%', 'max']

    max_toxicity = []
    max_toxicity_agg = []
    max_toxicity_diff = []
    max_toxicity_diff_agg = []
    for col in tqdm(score_cols):
        for context in contexts:
            for prompt in prompts:
                df_agg = perspective_scores[
                    (perspective_scores['prompt'] == prompt)
                    & (perspective_scores['context_type'] == context)
                ]
                df_agg = df_agg.assign(gen_index = ([0, 1, 2] * (df_agg.shape[0] // 3 + 1))[:df_agg.shape[0]])

                df_agg = pd.pivot_table(
                    df_agg, values=col, columns='model',
                    index=['ctf_set', 'gen_index', 'context', 'prompt'],
                ).reset_index()
                df_agg_max = df_agg.drop('context', axis=1).groupby(['ctf_set', 'gen_index', 'prompt']).agg('max')
                df_agg_min = df_agg.drop('context', axis=1).groupby(['ctf_set', 'gen_index', 'prompt']).agg('min')
                df_agg_diff = df_agg_max - df_agg_min

                df_agg = df_agg.assign(context=context, score=col)
                df_agg_diff = df_agg_diff.assign(context=context, score=col)

                max_toxicity.append(df_agg)
                max_toxicity_agg.append(format_df_agg(df_agg_max, agg_cols, col, context, prompt))
                max_toxicity_diff.append(df_agg_diff)
                max_toxicity_diff_agg.append(format_df_agg(df_agg_diff, agg_cols, col, context, prompt))

    return [
        pd.concat(max_toxicity),
        pd.concat(max_toxicity_agg),
        pd.concat(max_toxicity_diff),
        pd.concat(max_toxicity_diff_agg),
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perspective_dir",
        type=str,
        required=True,
        help="Path to the directory containing Perspective API outputs (one subdirectory per generation file)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path where the aggregated output CSVs will be written",
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    perspective_scores = load_perspective_scores(args.perspective_dir)
    max_toxicity, max_toxicity_agg, max_toxicity_diff, max_toxicity_diff_agg = calc_max_toxicity(perspective_scores)

    perspective_scores.to_csv(os.path.join(args.out_dir, 'perspective_scores.csv'))
    max_toxicity.to_csv(os.path.join(args.out_dir, 'max_toxicity.csv'))
    max_toxicity_agg.to_csv(os.path.join(args.out_dir, 'max_toxicity_agg.csv'))
    max_toxicity_diff.to_csv(os.path.join(args.out_dir, 'max_toxicity_diff.csv'))
    max_toxicity_diff_agg.to_csv(os.path.join(args.out_dir, 'max_toxicity_diff_agg.csv'))
