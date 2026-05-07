# Cultural Counterfactuals — Evaluation Code

This repository contains the scripts used to produce the LVLM bias-evaluation results in *Cultural Counterfactuals: Evaluating Cultural Bias in Large Vision-Language Models with Counterfactuals* (Howard, Su, & Fraser, 2026).

The dataset itself is hosted separately on the Hugging Face Hub at
[`thoughtworks/CulturalCounterfactuals`](https://huggingface.co/datasets/thoughtworks/CulturalCounterfactuals).

## Pipeline

```
       Cultural Counterfactuals images
    + metadata/<dimension>-post-filter.json
                  │
                  ▼
        ┌─────────────────────┐
        │     lvlm_gen.py     │   prompt one LVLM at a time over the dataset
        └─────────────────────┘
                  │   per-partition JSONL files
                  ▼
   ┌─────────────────────────────────┐
   │  concatenate_lvlm_outputs.py    │   merge partitions into one CSV per (model, prompt-set)
   └─────────────────────────────────┘
                  │   per-prefix CSVs
                  │
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
 ┌─────────────────┐   ┌──────────────────────────────┐
 │  perspective.py │   │  keywords_lexical_analysis.py│
 │                 │   │                              │
 │ Perspective API │   │ NRC-VAD + Warmth/Competence  │
 │ toxicity scores │   │ lexicons over the keyword    │
 │                 │   │ generations                  │
 └─────────────────┘   └──────────────────────────────┘
        │                    │
        ▼                    ▼
 ┌─────────────────┐    sensitivity / VAD / SCM
 │  lvlm_eval.py   │    tables + markdown report
 │                 │
 │ aggregate       │
 │ toxicity tables │
 └─────────────────┘
```

Each script is configured by command-line flags. The toxicity branch (`perspective.py` → `lvlm_eval.py`) and the lexical branch (`keywords_lexical_analysis.py`) are independent — run either or both, on any subset of dimensions (religion / nationality / socioeconomic).

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Hugging Face. Either grab a single dimension or the whole repo:

```bash
hf download thoughtworks/CulturalCounterfactuals \
    --repo-type dataset --local-dir cultural_counterfactuals_dataset
```

You will need:
- A GPU and the appropriate model weights (`lvlm_gen.py` downloads from the Hugging Face Hub on first use). One GPU per `lvlm_gen.py` worker is recommended.
- A [Perspective API](https://developers.perspectiveapi.com/) key (free tier works) for `perspective.py`.

## 1. Generate LVLM responses (`lvlm_gen.py`)

`lvlm_gen.py` iterates over the counterfactual sets in one dimension (`religion`, `nationality`, or `socioeconomic`), prompts a chosen LVLM with one or more questions, and appends each response to a JSONL file. Each row of the JSONL contains `{model, img_file_path, args, prompt, text}`.

Example — Qwen2.5-VL on the religion dimension, sampling 3 responses per (image, prompt):

```bash
python lvlm_gen.py \
    --ctf_dir cultural_counterfactuals_dataset/religion \
    --metadata cultural_counterfactuals_dataset/metadata/religion-post-filter.json \
    --out_dir output_lvlm_gen \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --batch_size 8 \
    --num_responses 3 \
    --prompts arrest,bad_influence,good_influence,should,shouldnt,keywords_v2 \
    --max_new_tokens 512
```

Supported models (the script branches on the model id):
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `google/gemma-3-12b-it`
- `OpenGVLab/InternVL3-{1B,8B,14B,38B}-hf`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `allenai/Molmo-7B-D-0924`

Each prompt key (`arrest`, `bad_influence`, `keywords_v2`, …) maps to a verbatim instruction defined in the `prompts` dict at the top of the script. Pass `--prompts /path/to/file.csv` to use a CSV with a `prompt` column instead.

### Optional flags

- `--people_only` — run on the source person images only (no cultural context background). Useful as a baseline.
- `--text_only` — run on a blank image for every prompt; provides a pure-text baseline.
- `--prompt_prefix "Answer briefly."` — prepend a string to every prompt.
- `--n_ctf_set 100` — randomly sample N counterfactual sets per prompt (for quick iteration).

### Parallelisation

Use `--n_partitions` and `--partition` to split the work across machines or GPUs. For example, to run 4 workers on a single 4-GPU node:

```bash
for p in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$p python lvlm_gen.py \
        --ctf_dir cultural_counterfactuals_dataset/religion \
        --metadata cultural_counterfactuals_dataset/metadata/religion-post-filter.json \
        --out_dir output_lvlm_gen \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --batch_size 8 --num_responses 3 \
        --prompts arrest,bad_influence,good_influence,should,shouldnt,keywords_v2 \
        --n_partitions 4 --partition $p &
done
wait
```

Each partition writes a separate JSONL named `<prefix>_<partition>.jsonl`, where `<prefix>` encodes the metadata file, model, prompts, and any flags.

## 2. Merge partition outputs (`concatenate_lvlm_outputs.py`)

After all partitions of a given run finish, concatenate them into one CSV per `<prefix>`:

```bash
python concatenate_lvlm_outputs.py --out_dir output_lvlm_gen
```

This finds every `<prefix>_<partition>.jsonl` in `--out_dir`, groups by prefix, and writes `output_lvlm_gen/<prefix>.csv`.

## 3. Score with Perspective (`perspective.py`)

For each merged generation CSV, score every generation using the Perspective API. The script handles model-specific text post-processing (e.g. stripping LLaVA's `[/INST]` prefix and Gemma's preamble paragraphs) so scores reflect the substantive answer.

```bash
python perspective.py \
    --out_dir output_perspective \
    --generations_file output_lvlm_gen/<prefix>.csv \
    --prompts arrest,bad_influence,good_influence,should,shouldnt,keywords_v2 \
    --api_key "$PERSPECTIVE_API_KEY"
```

Output structure:

```
output_perspective/
└── perspective-<prefix>/
    ├── perspective-<prefix>-0.jsonl
    └── perspective-<prefix>-0.csv
```

Output is resumable — if the JSONL already exists the script picks up where it left off.

### Parallelisation

Perspective rate-limits requests per API key. Split the workload across multiple keys or processes with `--n-partitions` and `--partition`:

```bash
for p in $(seq 0 9); do
    python perspective.py \
        --out_dir output_perspective \
        --generations_file output_lvlm_gen/<prefix>.csv \
        --prompts arrest,bad_influence,good_influence,should,shouldnt,keywords_v2 \
        --api_key "$PERSPECTIVE_API_KEY" \
        --n-partitions 10 --partition $p &
done
wait
```

Run `perspective.py` once per generations CSV (i.e., once per (model, dimension, prompt-set) configuration).

## 4. Aggregate toxicity (`lvlm_eval.py`)

Once Perspective has finished scoring every model you care about, aggregate the per-set toxicity statistics:

```bash
python lvlm_eval.py \
    --perspective_dir output_perspective \
    --out_dir output_eval
```

`lvlm_eval.py` walks every subdirectory of `--perspective_dir`, joins their CSVs into a single `perspective_scores` table, then for each `(prompt, context_type)` pair computes per-counterfactual-set:

- the **maximum** toxicity across models (`max_toxicity.csv`)
- the **(max − min)** spread across models (`max_toxicity_diff.csv`)

…together with their cross-set percentile summaries:

- `max_toxicity_agg.csv`
- `max_toxicity_diff_agg.csv`

The `_diff` tables capture how much toxicity *varies across models* on the same image, which is the headline measurement reported in the paper.

## 5. Lexical analysis of keyword generations (`keywords_lexical_analysis.py`)

For prompts that elicit a list of keywords (e.g. `keywords_v2`, `religious_moral_values`, `nationality_ethical_values`), `keywords_lexical_analysis.py` computes three lexical bias measures used in the paper's Lexical Analysis section:

1. **Jaccard sensitivity** — for each (counterfactual set × values_type × model), the average pairwise Jaccard overlap between the keyword sets generated under different cultural contexts; sensitivity is reported as `1 − mean Jaccard`.
2. **VAD** — proportion of unique keywords (per context × values_type × model) that fall above / below the NRC-VAD valence, arousal, and dominance thresholds.
3. **SCM** — proportion of unique keywords classified as high warmth (Sociability + Morality) or high competence (Ability + Agency) under the Warmth/Competence lexicon.

Results are written to a markdown report alongside three CSVs.

### Inputs

The script consumes a single CSV with one row per (image × prompt × seed) and the following columns:

| column          | description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `img_file_path` | path of the form `…/output/<dimension>/<set_id>/<set_id>_<context>.png`     |
| `model`         | HF model id                                                                 |
| `context`       | cultural context label (e.g. `Mosque`, `Brazil`, `low income`)              |
| `values_type`   | which keyword prompt produced this row (e.g. `moral`, `keywords_v2`)        |
| `values`        | a stringified Python list of keywords (output of the LVLM)                  |
| `refusal`       | boolean — rows where the model refused to answer are filtered out           |
| `gender`, `age`, `race` | parsed demographic attributes of the depicted person                |

This format is **not** produced directly by `concatenate_lvlm_outputs.py`. Producing it is a small preprocessing step: filter the merged CSVs to keyword-style prompts, parse the LVLM's `text` field into a Python list of keywords, attach a `refusal` flag, and join the demographic attributes from the counterfactual-set id.

You will also need the two lexicons used by the paper:

- **NRC-VAD v2.1** — [Mohammad, 2025](https://arxiv.org/abs/2503.23547); download `NRC-VAD-Lexicon-v2.1.txt` (tab-separated `term`, `valence`, `arousal`, `dominance`).
- **Warmth/Competence (SCM) lexicon** — Nicolas, Bai & Fiske, 2021; CSV with `Sociability_dict`/`Morality_dict`/`Ability_dict`/`Agency_dict`/`Status_dict` flags and `*_dir` direction columns.

### Run

```bash
python keywords_lexical_analysis.py \
    --input model_outputs/religion-values-prompts.csv \
    --vad-lexicon NRC-VAD-Lexicon-v2.1.txt \
    --scm-lexicon warmth_competence_lexicon.csv \
    --report output_lexical/religion_report.md
```

### Outputs

Given `--report path/to/<name>.md`, the script writes:

- `path/to/<name>.md` — markdown report with summary tables and LaTeX-ready versions of each table
- `path/to/<name>_sensitivity.csv` — per-(model, values_type) Jaccard sensitivity stats
- `path/to/<name>_vad.csv` — per-(model, context, values_type) VAD extreme-value proportions
- `path/to/<name>_scm.csv` — per-(model, context, values_type) high-warmth / high-competence proportions

## Citation

```bibtex
@article{howard2026cultural,
  title={Cultural Counterfactuals: Evaluating Cultural Biases in Large Vision-Language Models with Counterfactual Examples},
  author={Howard, Phillip and Su, Xin and Fraser, Kathleen C},
  journal={arXiv preprint arXiv:2603.02370},
  year={2026}
}
```
