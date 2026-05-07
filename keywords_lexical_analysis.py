import argparse
import ast
import itertools
import math
import re
from pathlib import Path

import pandas as pd


USECOLS = [
    "img_file_path",
    "context",
    "values_type",
    "values",
    "refusal",
    "model",
    "gender",
    "age",
    "race",
]

_SET_ID_RE = re.compile(r"/output/[^/]+/([^/]+)/")

SCM_BASE_DIMS = [
    ("sociability", "Sociability_dict", "Sociability_dir"),
    ("morality", "Morality_dict", "Morality_dir"),
    ("ability", "Ability_dict", "Ability_dir"),
    ("agency", "Agency_dict", "Agency_dir"),
    ("status", "Status_dict", "Status_dir"),
]

VAD_THRESHOLDS = {
    "pos_valence": 0.5,
    "neg_valence": -0.5,
    "high_arousal": 0.75,
    "low_arousal": -0.75,
    "high_dominance": 0.75,
    "low_dominance": -0.75,
}


def clean_keyword(kw):
    """Normalize a keyword: lowercase, strip punctuation, collapse whitespace."""
    kw = (kw or "").lower().strip()
    kw = re.sub(r"[^\w\s-]", "", kw)
    kw = re.sub(r"\s+", " ", kw)
    return kw.strip()


def parse_values(raw):
    """Parse the string-repr values list into a cleaned keyword list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [clean_keyword(str(v)) for v in parsed if v]


def extract_set_id(path):
    """Extract the person set_id from an img_file_path."""
    m = _SET_ID_RE.search(path)
    return m.group(1) if m else None


def aggregate_values(df):
    """Union keywords across seeds per (set_id, values_type, context, model)."""
    grouped = df.groupby(["set_id", "values_type", "context", "model"])["values_clean"].apply(
        lambda lists: set().union(*lists)
    ).reset_index()
    grouped.rename(columns={"values_clean": "values_set"}, inplace=True)
    print(f"[AGG] Groups: {len(grouped):,}")
    return grouped


def jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_sensitivity(agg):
    """Compute sensitivity = 1 - avg pairwise Jaccard per (set_id, values_type, model)."""
    rows = []
    for (set_id, vtype, model), group in agg.groupby(["set_id", "values_type", "model"]):
        sets_by_ctx = dict(zip(group["context"], group["values_set"]))
        if len(sets_by_ctx) < 2:
            continue
        sims = [jaccard(sets_by_ctx[a], sets_by_ctx[b])
                for a, b in itertools.combinations(sets_by_ctx, 2)]
        avg_j = sum(sims) / len(sims)
        rows.append({"set_id": set_id, "values_type": vtype, "model": model,
                      "avg_jaccard": avg_j, "sensitivity": 1 - avg_j,
                      "n_contexts": len(sets_by_ctx)})
    out = pd.DataFrame(rows)
    print(f"[SENSITIVITY] Sets computed: {len(out):,}")
    return out


def load_vad_lexicon(path):
    """Load NRC-VAD lexicon into a term -> (valence, arousal, dominance) dict."""
    df = pd.read_csv(path, sep="\t")
    out = {}
    for _, row in df.iterrows():
        term = clean_keyword(str(row["term"]))
        if term and term not in out:
            out[term] = (float(row["valence"]), float(row["arousal"]), float(row["dominance"]))
    print(f"[VAD] Lexicon loaded: {len(out):,} terms")
    return out


def vad_match(kw, vad_map):
    """Match a keyword to VAD scores: phrase first, then token fallback."""
    phrase = clean_keyword(kw)
    if not phrase:
        return []
    hit = vad_map.get(phrase)
    if hit is not None:
        return [hit]
    out = []
    for tok in phrase.split():
        h = vad_map.get(tok)
        if h is not None:
            out.append(h)
    return out


def compute_vad(agg, vad_map):
    """VAD extreme-value proportions on unique vocabulary per (context, values_type, model)."""
    rows = []
    for (ctx, vtype, model), group in agg.groupby(["context", "values_type", "model"]):
        unique_terms = set()
        for kw_set in group["values_set"]:
            unique_terms.update(kw_set)

        n_matched = 0
        counts = {k: 0 for k in VAD_THRESHOLDS}
        for kw in unique_terms:
            scores = vad_match(kw, vad_map)
            if not scores:
                continue
            n_matched += 1
            vals = [s[0] for s in scores]
            aros = [s[1] for s in scores]
            doms = [s[2] for s in scores]
            if any(v > VAD_THRESHOLDS["pos_valence"] for v in vals):
                counts["pos_valence"] += 1
            if any(v < VAD_THRESHOLDS["neg_valence"] for v in vals):
                counts["neg_valence"] += 1
            if any(a > VAD_THRESHOLDS["high_arousal"] for a in aros):
                counts["high_arousal"] += 1
            if any(a < VAD_THRESHOLDS["low_arousal"] for a in aros):
                counts["low_arousal"] += 1
            if any(d > VAD_THRESHOLDS["high_dominance"] for d in doms):
                counts["high_dominance"] += 1
            if any(d < VAD_THRESHOLDS["low_dominance"] for d in doms):
                counts["low_dominance"] += 1

        row = {"context": ctx, "values_type": vtype, "model": model,
               "n_terms_unique": len(unique_terms), "n_matched": n_matched}
        for k, c in counts.items():
            row[f"pct_{k}"] = c / n_matched if n_matched else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"[VAD] Groups computed: {len(out):,}")
    return out


def load_scm_lexicon(path):
    """Load SCM lexicon into dim_name -> {term: sign} mappings."""
    df = pd.read_csv(path)
    term_cols = [c for c in df.columns if c.startswith("values")]
    dim_to_term_dir = {dim: {} for dim, _, _ in SCM_BASE_DIMS}
    dim_conflicts = {dim: set() for dim, _, _ in SCM_BASE_DIMS}

    for _, row in df.iterrows():
        terms = []
        for c in term_cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            t = clean_keyword(str(v))
            if t:
                terms.append(t)
        if not terms:
            continue

        for dim, dict_col, dir_col in SCM_BASE_DIMS:
            try:
                is_in = int(row.get(dict_col, 0)) == 1
            except Exception:
                is_in = False
            if not is_in:
                continue
            try:
                d = float(row.get(dir_col))
            except Exception:
                continue
            sign = 1 if d > 0 else (-1 if d < 0 else 0)
            if sign == 0:
                continue

            m = dim_to_term_dir[dim]
            conflicts = dim_conflicts[dim]
            for t in terms:
                if t in conflicts:
                    continue
                if t not in m:
                    m[t] = sign
                elif m[t] != sign:
                    conflicts.add(t)
                    m.pop(t, None)

    total = sum(len(v) for v in dim_to_term_dir.values())
    print(f"[SCM] Lexicon loaded: {total:,} term-dimension entries")
    return dim_to_term_dir


def scm_item_signs(kw, dim_to_term_dir):
    """Score a keyword against SCM subdimensions via token matching."""
    phrase = clean_keyword(kw)
    if not phrase:
        return {dim: None for dim, _, _ in SCM_BASE_DIMS}
    tokens = [t for t in phrase.split() if t]
    out = {}
    for dim, _, _ in SCM_BASE_DIMS:
        lookup = dim_to_term_dir.get(dim, {})
        signs = {lookup[t] for t in tokens if t in lookup}
        if not signs:
            out[dim] = None
        elif signs == {1}:
            out[dim] = 1
        elif signs == {-1}:
            out[dim] = -1
        else:
            out[dim] = 0
    return out


def combine_sign(signs):
    """Combine multiple signed matches into a single sign (+1/-1/0/None)."""
    hits = [s for s in signs if s is not None]
    if not hits:
        return None
    non_zero = {s for s in hits if s != 0}
    if len(non_zero) >= 2:
        return 0
    if len(non_zero) == 1:
        return next(iter(non_zero))
    return 0


def compute_scm(agg, dim_to_term_dir):
    """SCM warmth/competence proportions on unique vocabulary per (context, values_type, model)."""
    rows = []
    for (ctx, vtype, model), group in agg.groupby(["context", "values_type", "model"]):
        unique_terms = set()
        for kw_set in group["values_set"]:
            unique_terms.update(kw_set)

        n_matched = 0
        high_warmth = 0
        high_competence = 0
        for kw in unique_terms:
            base = scm_item_signs(kw, dim_to_term_dir)
            warmth_sign = combine_sign([base.get("sociability"), base.get("morality")])
            competence_sign = combine_sign([base.get("ability"), base.get("agency")])
            any_hit = warmth_sign is not None or competence_sign is not None or base.get("status") is not None
            if not any_hit:
                continue
            n_matched += 1
            if warmth_sign == 1:
                high_warmth += 1
            if competence_sign == 1:
                high_competence += 1

        rows.append({
            "context": ctx,
            "values_type": vtype,
            "model": model,
            "n_terms_unique": len(unique_terms),
            "n_matched": n_matched,
            "pct_high_warmth": high_warmth / n_matched if n_matched else float("nan"),
            "pct_high_competence": high_competence / n_matched if n_matched else float("nan"),
        })

    out = pd.DataFrame(rows)
    print(f"[SCM] Groups computed: {len(out):,}")
    return out


def write_report(
    out_path,
    *,
    input_path,
    n_rows,
    n_refusals,
    n_after_filter,
    n_valid,
    n_set_ids,
    contexts,
    values_types,
    models,
    n_agg_groups,
    sens,
    vad,
    scm,
):
    """Write all analysis results to a markdown report."""
    lines = []

    lines.append("# Cultural Counterfactuals — Lexical Analysis Report\n")

    lines.append("## Data Summary\n")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Rows read: {n_rows:,}")
    lines.append(f"- Refusals: {n_refusals:,} ({n_refusals/n_rows*100:.1f}%)")
    lines.append(f"- Rows after refusal filter: {n_after_filter:,}")
    lines.append(f"- Rows with valid parsed values: {n_valid:,}")
    lines.append(f"- Unique set_ids (persons): {n_set_ids:,}")
    lines.append(f"- Models: {', '.join(models)}")
    lines.append(f"- Contexts: {', '.join(contexts)}")
    lines.append(f"- Values types: {', '.join(values_types)}")
    lines.append(f"- Aggregated groups (set_id x values_type x context): {n_agg_groups:,}")
    lines.append("")

    lines.append("## Jaccard Keyword Sensitivity\n")
    lines.append(f"- Sets computed: {len(sens):,}")
    lines.append(f"- Sensitivity = 1 - avg pairwise Jaccard across contexts\n")
    sens_summary = sens.groupby(["model", "values_type"])["sensitivity"].agg(["count", "mean", "std", "min", "median", "max"])
    lines.append("| model | values_type | count | mean | std | min | median | max |")
    lines.append("|-------|-------------|-------|------|-----|-----|--------|-----|")
    for (model, vtype), row in sens_summary.iterrows():
        lines.append(f"| {model} | {vtype} | {int(row['count'])} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['median']:.4f} | {row['max']:.4f} |")
    lines.append("")

    lines.append("### Jaccard Sensitivity LaTeX\n")
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\hline")
    lines.append(r"Values Type & Model & Mean & Std \\")
    lines.append(r"\hline")
    for _, row in sens_summary.reset_index().sort_values(["values_type", "model"]).iterrows():
        short_model = row["model"].split("/", 1)[-1]
        lines.append(f"{row['values_type']} & {short_model} & {row['mean']:.2f} & {row['std']:.2f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Jaccard Keyword Sensitivity by Model and Values Type}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    lines.append("## VAD Analysis (NRC-VAD Lexicon)\n")
    lines.append(f"- Thresholds: valence +/-0.5, arousal +/-0.75, dominance +/-0.75")
    lines.append(f"- Unit: unique vocabulary per (context, values_type)\n")
    lines.append("| model | context | values_type | n_terms_unique | n_matched | pct_pos_valence | pct_neg_valence | pct_high_arousal | pct_low_arousal | pct_high_dominance | pct_low_dominance |")
    lines.append("|-------|---------|-------------|---------------|-----------|-----------------|-----------------|------------------|-----------------|--------------------|--------------------|")
    for _, row in vad.iterrows():
        lines.append(f"| {row['model']} | {row['context']} | {row['values_type']} | {row['n_terms_unique']:,} | {row['n_matched']:,} | {row['pct_pos_valence']:.4f} | {row['pct_neg_valence']:.4f} | {row['pct_high_arousal']:.4f} | {row['pct_low_arousal']:.4f} | {row['pct_high_dominance']:.4f} | {row['pct_low_dominance']:.4f} |")
    lines.append("")

    lines.append("### VAD Analysis LaTeX\n")
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lllrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Values Type & Model & Context & Pos Valence & Neg Valence & High Arousal & Low Arousal & High Dominance & Low Dominance \\")
    lines.append(r"\hline")
    for _, row in vad.sort_values(["values_type", "model"]).iterrows():
        short_model = row["model"].split("/", 1)[-1]
        lines.append(f"{row['values_type']} & {short_model} & {row['context']} & {row['pct_pos_valence']:.2f} & {row['pct_neg_valence']:.2f} & {row['pct_high_arousal']:.2f} & {row['pct_low_arousal']:.2f} & {row['pct_high_dominance']:.2f} & {row['pct_low_dominance']:.2f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{VAD Analysis (NRC-VAD Lexicon) by Model, Context, and Values Type}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    lines.append("## SCM Analysis (Warmth / Competence)\n")
    lines.append(f"- Warmth = Sociability + Morality")
    lines.append(f"- Competence = Ability + Agency")
    lines.append(f"- Unit: unique vocabulary per (context, values_type)\n")
    lines.append("| model | context | values_type | n_terms_unique | n_matched | pct_high_warmth | pct_high_competence |")
    lines.append("|-------|---------|-------------|---------------|-----------|-----------------|---------------------|")
    for _, row in scm.iterrows():
        lines.append(f"| {row['model']} | {row['context']} | {row['values_type']} | {row['n_terms_unique']:,} | {row['n_matched']:,} | {row['pct_high_warmth']:.4f} | {row['pct_high_competence']:.4f} |")
    lines.append("")

    lines.append("### SCM Analysis LaTeX\n")
    scm_vtypes = sorted(scm["values_type"].unique())
    scm_pivoted = scm.copy()
    scm_pivoted["short_model"] = scm_pivoted["model"].str.split("/", n=1).str[-1]
    scm_pivoted = (
        scm_pivoted
        .set_index(["short_model", "context", "values_type"])[["pct_high_warmth", "pct_high_competence"]]
        .unstack("values_type")
        .sort_index()
    )
    col_spec = "ll" + "rr" * len(scm_vtypes)
    lines.append("```latex")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    vtype_header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{vt}}}" for vt in scm_vtypes)
    lines.append(f"Model & Context & {vtype_header} \\\\")
    metric_header = " & ".join(["High Warmth & High Competence"] * len(scm_vtypes))
    lines.append(f" & & {metric_header} \\\\")
    lines.append(r"\hline")
    for (short_model, context), row in scm_pivoted.iterrows():
        cells = " & ".join(
            f"{row[('pct_high_warmth', vt)]:.2f} & {row[('pct_high_competence', vt)]:.2f}"
            for vt in scm_vtypes
        )
        lines.append(f"{short_model} & {context} & {cells} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{SCM Analysis (Warmth / Competence) by Model and Context}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[REPORT] Written to {out_path}")

    base = out_path.parent / out_path.stem
    sens_summary.reset_index().to_csv(f"{base}_sensitivity.csv", index=False)
    print(f"[REPORT] Written to {base}_sensitivity.csv")
    vad.to_csv(f"{base}_vad.csv", index=False)
    print(f"[REPORT] Written to {base}_vad.csv")
    scm.to_csv(f"{base}_scm.csv", index=False)
    print(f"[REPORT] Written to {base}_scm.csv")


def load_data(csv_path):
    """Load CSV, filter refusals, parse values, extract set_id."""
    df = pd.read_csv(csv_path, usecols=USECOLS)
    print(f"[LOAD] Rows read: {len(df):,}")
    print(f"[LOAD] Refusals: {df['refusal'].sum():,}")
    df = df[df["refusal"] == False].drop(columns=["refusal"])
    print(f"[LOAD] Rows after filtering: {len(df):,}")
    df["values_clean"] = df["values"].apply(parse_values)
    df = df[df["values_clean"].map(len) > 0]
    print(f"[LOAD] Rows with valid values: {len(df):,}")
    df["set_id"] = df["img_file_path"].apply(extract_set_id)
    missing_set_id = df["set_id"].isna().sum()
    if missing_set_id:
        print(f"[WARN] Rows with no set_id: {missing_set_id:,}")
        df = df.dropna(subset=["set_id"])
    print(f"[LOAD] Unique set_ids: {df['set_id'].nunique():,}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("model_outputs/religion-values-prompts.csv"))
    parser.add_argument("--vad-lexicon", type=Path, default=Path("artifacts/data/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"))
    parser.add_argument("--scm-lexicon", type=Path, default=Path("artifacts/data/warmth_competence_lexicon.csv"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/outputs/lexical_analysis_report.md"))
    args = parser.parse_args()

    raw = pd.read_csv(args.input, usecols=USECOLS)
    n_rows = len(raw)
    n_refusals = int(raw["refusal"].sum())
    contexts = sorted(raw["context"].dropna().unique())
    values_types = sorted(raw["values_type"].dropna().unique())
    models = sorted(raw["model"].dropna().unique())
    del raw

    df = load_data(args.input)
    n_after_filter = len(df) + (n_rows - n_refusals - len(df))  # before values parse
    n_valid = len(df)
    n_set_ids = df["set_id"].nunique()

    agg = aggregate_values(df)

    sens = compute_sensitivity(agg)
    print(sens.head(10))
    print(f"\nSensitivity by model and values_type:")
    print(sens.groupby(["model", "values_type"])["sensitivity"].describe())

    vad_map = load_vad_lexicon(args.vad_lexicon)
    vad = compute_vad(agg, vad_map)
    print(f"\nVAD by context and values_type:")
    print(vad.to_string())

    scm_lex = load_scm_lexicon(args.scm_lexicon)
    scm = compute_scm(agg, scm_lex)
    print(f"\nSCM by context and values_type:")
    print(scm.to_string())

    write_report(
        args.report,
        input_path=args.input,
        n_rows=n_rows,
        n_refusals=n_refusals,
        n_after_filter=n_rows - n_refusals,
        n_valid=n_valid,
        n_set_ids=n_set_ids,
        contexts=contexts,
        values_types=values_types,
        models=models,
        n_agg_groups=len(agg),
        sens=sens,
        vad=vad,
        scm=scm,
    )
