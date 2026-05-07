"""Concatenate per-partition JSONL outputs from lvlm_gen.py into a single CSV
per (model, prompt-set, etc.) configuration.

lvlm_gen.py writes one JSONL per partition with a filename pattern
`<prefix>_<partition>.jsonl`; this script groups files by their `<prefix>`,
concatenates their rows, and writes one consolidated CSV per prefix.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# Number of partitions used in the canonical run for the InternVL3 model
# variants. Re-running with a different `--n_partitions` value sometimes
# produced overlapping outputs; we keep only the rows from the canonical run
# to avoid duplicate generations in downstream analyses.
INTERNVL_CANONICAL_N_PARTITIONS = {
    'InternVL3-38B-hf': 6,
    'InternVL3-1B-hf': 4,
    'InternVL3-14B-hf': 4,
}


def concatenate(out_dir):
    """Concatenate every set of `<prefix>_<partition>.jsonl` files in
    `out_dir` and write `<prefix>.csv` next to them."""
    out_dir = Path(out_dir)
    prefixes = {
        '_'.join(p.stem.split('_')[:-1])
        for p in out_dir.iterdir()
        if p.suffix == '.jsonl'
    }

    for prefix in prefixes:
        files = sorted(out_dir.glob(f'{prefix}_*.jsonl'))
        rows = []
        for file in files:
            with open(file, 'r') as f:
                rows.extend(json.loads(line) for line in f)
        df = pd.DataFrame(rows)

        # Filter out duplicate generations from non-canonical InternVL3 runs.
        for model_tag, n_parts in INTERNVL_CANONICAL_N_PARTITIONS.items():
            if model_tag in prefix:
                df = df.assign(n_partitions=df['args'].apply(lambda x: x['n_partitions']))
                df = df[df['n_partitions'] == n_parts]
                break

        df.to_csv(out_dir / f'{prefix}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='Directory containing the per-partition JSONL outputs from lvlm_gen.py. '
             'Consolidated CSVs are written alongside them.',
    )
    args = parser.parse_args()
    concatenate(args.out_dir)
