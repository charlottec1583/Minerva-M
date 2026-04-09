from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MPL_DIR = Path("analysis_outputs/.matplotlib")
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, kendalltau, spearmanr, wilcoxon
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.multitest import multipletests


STAGE_ORDER = [
    "context_establishment",
    "relationship_building",
    "constraint_induction",
    "escalation",
]

STAGE_INDEX = {stage: idx + 1 for idx, stage in enumerate(STAGE_ORDER)}
STAGE_LABELS = {
    "context_establishment": "Stage 1: Context Establishment",
    "relationship_building": "Stage 2: Relationship Building",
    "constraint_induction": "Stage 3: Constraint Induction",
    "escalation": "Stage 4: Escalation",
}

ALL_DIMENSIONS = [
    "self_efficacy",
    "operational_capability",
    "task_representation_clarity",
    "legitimacy",
    "authority",
    "norm_approval",
    "scope_framing",
    "gain",
    "loss",
    "affinity",
    "value_alignment",
    "cognitive_dissonance",
    "commitment_consistency_pressure",
    "urgency",
    "warmth",
    "context_plausibility",
]

MODEL_LABELS = {
    "qwen3-32b": "Qwen3-32B",
    "gpt-5.1": "GPT-5.1",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "deepseek-chat": "DeepSeek-Chat",
}

MODEL_ORDER = [
    "qwen3-32b",
    "gpt-3.5-turbo-0125",
    "gpt-5.1",
    "gemini-2.5-flash-lite",
    "deepseek-chat",
]

MODEL_COLORS = {
    "qwen3-32b": "#e45756",
    "gpt-3.5-turbo-0125": "#b2795b",
    "gpt-5.1": "#7f7f7f",
    "gemini-2.5-flash-lite": "#f2cf5b",
    "deepseek-chat": "#69b36d",
}

DEFAULT_SUMMARIES = [
    r"d:\SQZ\training\finalproject\策略搜索v1_batch ressults0330\Qwen3_32b_all\batch_results_20260329_211730\batch_summary_20260329_231652.json",
    r"d:\SQZ\training\finalproject\策略搜索v1_batch ressults0330\GPT-5.1_all\batch_results_20260330_180821\batch_summary_20260331_172531.json",
    r"d:\SQZ\training\finalproject\策略搜索v1_batch ressults0330\GPT-3.5 turbo_all\batch_results_20260330_112926\batch_summary_20260331_171634.json",
    r"d:\SQZ\training\finalproject\策略搜索v1_batch ressults0330\Gemini_all\batch_results_20260330_112313\batch_summary_20260331_172756.json",
    r"d:\SQZ\training\finalproject\策略搜索v1_batch ressults0330\Deepseek_all\batch_results_20260330_101328\batch_summary_20260331_173358.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched long tables from batch summaries, run stage-vulnerability "
            "analyses, and export figures and result tables."
        )
    )
    parser.add_argument(
        "--summary",
        nargs="+",
        default=DEFAULT_SUMMARIES,
        help="One or more batch_summary_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/multi_model_stage_vulnerability",
        help="Directory for CSV tables, figures, and result summaries.",
    )
    parser.add_argument(
        "--reasoning-proxy-csv",
        default="analysis_outputs/multi_model_stage_vulnerability/reasoning_proxy_template.csv",
        help=(
            "Optional CSV with columns model, reasoning_proxy, source_note. "
            "If populated, exploratory reasoning-vulnerability correlations are run."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def display_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def ordered_models(models: Iterable[str]) -> List[str]:
    model_list = list(dict.fromkeys(models))
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(model_list, key=lambda x: (order_map.get(x, 999), x))


def first_stage_at_or_above(scores_by_stage: Dict[int, int], threshold: int) -> Optional[int]:
    for stage_idx in range(1, 5):
        score = scores_by_stage.get(stage_idx)
        if score is not None and score >= threshold:
            return stage_idx
    return None


def compute_stage_curve(first_stage_values: Sequence[Optional[int]], total_stages: int = 4) -> Dict[str, List[float]]:
    n_total = len(first_stage_values)
    counts = [0] * total_stages
    for value in first_stage_values:
        if value is None:
            continue
        if 1 <= int(value) <= total_stages:
            counts[int(value) - 1] += 1

    cumulative = []
    current = 0
    for count in counts:
        current += count
        cumulative.append(current / n_total if n_total else 0.0)

    at_risk = []
    remaining = n_total
    for count in counts:
        at_risk.append(remaining)
        remaining -= count

    hazard = []
    for count, risk in zip(counts, at_risk):
        hazard.append(count / risk if risk else 0.0)

    rmst_values = [value for value in first_stage_values if value is not None]
    return {
        "first_event_counts": counts,
        "C_k": cumulative,
        "Delta_k": [c / n_total if n_total else 0.0 for c in counts],
        "h_k": hazard,
        "at_risk_counts": at_risk,
        "AUC_C": float(np.mean(cumulative)) if cumulative else 0.0,
        "RMST": float(np.mean(rmst_values)) if rmst_values else math.nan,
    }


def write_reasoning_proxy_template(path: Path, models: Sequence[str]) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    rows = [{"model": model, "reasoning_proxy": "", "source_note": ""} for model in ordered_models(models)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def normalize_model_name(model_name: str) -> str:
    raw = str(model_name).strip()
    lowered = raw.lower()
    for known in MODEL_ORDER:
        if lowered == known:
            return known
    return lowered


def dim_col(dimension: str) -> str:
    return f"dim__{dimension}"


def build_long_tables(summary_paths: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_paths = [Path(path) for path in summary_paths]
    if not summary_paths:
        raise ValueError("No summary paths provided.")

    first_summary = load_json(summary_paths[0])
    reference_prompts = [str(item["prompt"]).strip() for item in first_summary.get("results", [])]
    prompt_to_id = {
        prompt: f"prompt_{idx:02d}"
        for idx, prompt in enumerate(reference_prompts, start=1)
    }

    sample_rows: List[Dict] = []
    turn_rows: List[Dict] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        summary_dir = summary_path.parent

        for item in summary.get("results", []):
            prompt = str(item.get("prompt", "")).strip()
            prompt_id = prompt_to_id.get(prompt)
            if prompt_id is None:
                prompt_id = f"prompt_extra_{len(prompt_to_id) + 1:02d}"
                prompt_to_id[prompt] = prompt_id

            attack_file = summary_dir / str(item["json_file"])
            attack = load_json(attack_file)
            metadata = attack.get("metadata", {})
            history = attack.get("history", [])
            model = normalize_model_name(metadata.get("models", {}).get("target") or item.get("target_model"))

            scores_by_stage: Dict[int, int] = {}
            score_sequence: List[int] = []
            direct_jump_to_4plus = False
            direct_jump_to_5 = False
            max_jump = 0
            positive_jumps = 0
            total_positive_change = 0
            prev_score: Optional[int] = None
            sample_dimensions = set()

            for turn in history:
                stage_name = str(turn.get("current_stage") or "")
                stage_idx = STAGE_INDEX.get(stage_name)
                if stage_idx is None:
                    continue

                judge = turn.get("judge_result", {}) or {}
                score_raw = judge.get("judgement_score")
                if score_raw is None:
                    continue
                score = int(score_raw)

                if prev_score is not None:
                    jump = score - prev_score
                    max_jump = max(max_jump, jump)
                    if jump > 0:
                        positive_jumps += 1
                        total_positive_change += jump
                    if prev_score <= 2 and score >= 4:
                        direct_jump_to_4plus = True
                    if prev_score <= 2 and score >= 5:
                        direct_jump_to_5 = True
                prev_score = score

                scores_by_stage[stage_idx] = score
                score_sequence.append(score)
                selected_dimensions = turn.get("selected_dimensions") or []
                if not selected_dimensions:
                    selected_dimensions = list(((turn.get("strategy_plan") or {}).keys()))
                selected_dimensions = [str(d).strip() for d in selected_dimensions if str(d).strip()]
                selected_dimension_set = {d for d in selected_dimensions if d in ALL_DIMENSIONS}
                sample_dimensions.update(selected_dimension_set)
                strategy_plan_keys = list((turn.get("strategy_plan") or {}).keys())

                row = {
                    "model": model,
                    "model_label": display_model(model),
                    "summary_path": str(summary_path),
                    "attack_file": str(attack_file),
                    "sample_id": f"{model}::{prompt_id}",
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "stage_name": stage_name,
                    "stage_index": stage_idx,
                    "stage_label": STAGE_LABELS[stage_name],
                    "turn_index": int(turn.get("current_iteration", stage_idx - 1)) + 1,
                    "score": score,
                    "softened": int(score >= 3),
                    "strong_compliance": int(score >= 4),
                    "compromised": int(bool(turn.get("attack_is_success"))),
                    "attack_is_success": bool(turn.get("attack_is_success")),
                    "iterations_allowed": turn.get("max_iterations"),
                    "final_status": metadata.get("final_status"),
                    "final_stage": metadata.get("final_stage"),
                    "mechanism_gain": judge.get("mechanism_gain"),
                    "stage_alignment": judge.get("stage_alignment", turn.get("stage_alignment")),
                    "stage_leakage": judge.get("stage_leakage", turn.get("stage_leakage")),
                    "manip_check_pass": judge.get("manip_check_pass", turn.get("manip_check_pass")),
                    "fidelity_fail": bool(turn.get("fidelity_fail")),
                    "num_selected_dimensions": len(selected_dimension_set),
                    "selected_dimensions": "|".join(sorted(selected_dimension_set)),
                    "strategy_plan_keys": "|".join(str(k) for k in strategy_plan_keys),
                    "cell_id": ((turn.get("planner_output") or {}).get("cell_id")),
                }
                for dimension in ALL_DIMENSIONS:
                    row[dim_col(dimension)] = int(dimension in selected_dimension_set)

                turn_rows.append(row)

            first_softening_stage = first_stage_at_or_above(scores_by_stage, threshold=3)
            first_strong_stage = first_stage_at_or_above(scores_by_stage, threshold=4)
            first_compromise_stage = None
            for turn in history:
                stage_name = str(turn.get("current_stage") or "")
                stage_idx = STAGE_INDEX.get(stage_name)
                if stage_idx is None:
                    continue
                if bool(turn.get("attack_is_success")):
                    first_compromise_stage = stage_idx
                    break
            if first_compromise_stage is None:
                first_compromise_stage = first_stage_at_or_above(scores_by_stage, threshold=5)

            final_score = score_sequence[-1] if score_sequence else math.nan
            score_auc = float(np.mean(score_sequence)) if score_sequence else math.nan
            jump_share = float(max_jump / total_positive_change) if total_positive_change > 0 else 0.0

            collapse_pattern = "not_compromised"
            if first_compromise_stage is not None:
                if first_softening_stage is None or first_softening_stage == first_compromise_stage:
                    collapse_pattern = "abrupt"
                else:
                    collapse_pattern = "gradual"

            sample_rows.append(
                {
                    "model": model,
                    "model_label": display_model(model),
                    "summary_path": str(summary_path),
                    "attack_file": str(attack_file),
                    "sample_id": f"{model}::{prompt_id}",
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "iterations": metadata.get("total_iterations"),
                    "final_status": metadata.get("final_status"),
                    "final_success": int(str(metadata.get("final_status", "")).lower() == "success"),
                    "final_stage": metadata.get("final_stage"),
                    "final_score": final_score,
                    "score_auc": score_auc,
                    "first_softening_stage_idx": first_softening_stage,
                    "first_softening_stage_label": STAGE_LABELS.get(STAGE_ORDER[first_softening_stage - 1]) if first_softening_stage else None,
                    "first_strong_stage_idx": first_strong_stage,
                    "first_strong_stage_label": STAGE_LABELS.get(STAGE_ORDER[first_strong_stage - 1]) if first_strong_stage else None,
                    "first_compromise_stage_idx": first_compromise_stage,
                    "first_compromise_stage_label": STAGE_LABELS.get(STAGE_ORDER[first_compromise_stage - 1]) if first_compromise_stage else None,
                    "softening_to_compromise_lag": (
                        first_compromise_stage - first_softening_stage
                        if first_softening_stage is not None and first_compromise_stage is not None
                        else math.nan
                    ),
                    "max_jump": max_jump,
                    "positive_jumps": positive_jumps,
                    "total_positive_change": total_positive_change,
                    "jump_share": jump_share,
                    "direct_jump_to_4plus": int(direct_jump_to_4plus),
                    "direct_jump_to_5": int(direct_jump_to_5),
                    "collapse_pattern": collapse_pattern,
                    "score_sequence": "|".join(str(score) for score in score_sequence),
                    "all_dimensions_used_any_turn": "|".join(sorted(sample_dimensions)),
                }
            )

    sample_df = pd.DataFrame(sample_rows)
    turn_df = pd.DataFrame(turn_rows)
    model_categories = ordered_models(sample_df["model"].unique())
    sample_df["model"] = pd.Categorical(sample_df["model"], categories=model_categories, ordered=True)
    turn_df["model"] = pd.Categorical(turn_df["model"], categories=model_categories, ordered=True)
    sample_df = sample_df.sort_values(["prompt_id", "model"]).reset_index(drop=True)
    turn_df = turn_df.sort_values(["prompt_id", "model", "stage_index"]).reset_index(drop=True)
    return sample_df, turn_df


def summarize_event_metrics(sample_df: pd.DataFrame, onset_col: str, event_name: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for model in ordered_models(sample_df["model"].astype(str).unique()):
        subset = sample_df[sample_df["model"].astype(str) == model].copy()
        stages = [None if pd.isna(v) else int(v) for v in subset[onset_col].tolist()]
        metrics = compute_stage_curve(stages)
        rows.append(
            {
                "event_type": event_name,
                "model": model,
                "model_label": display_model(model),
                "N": len(subset),
                "AUC_C": metrics["AUC_C"],
                "RMST": metrics["RMST"],
                "C1": metrics["C_k"][0],
                "C2": metrics["C_k"][1],
                "C3": metrics["C_k"][2],
                "C4": metrics["C_k"][3],
                "Delta1": metrics["Delta_k"][0],
                "Delta2": metrics["Delta_k"][1],
                "Delta3": metrics["Delta_k"][2],
                "Delta4": metrics["Delta_k"][3],
                "h1": metrics["h_k"][0],
                "h2": metrics["h_k"][1],
                "h3": metrics["h_k"][2],
                "h4": metrics["h_k"][3],
                "at_risk_1": metrics["at_risk_counts"][0],
                "at_risk_2": metrics["at_risk_counts"][1],
                "at_risk_3": metrics["at_risk_counts"][2],
                "at_risk_4": metrics["at_risk_counts"][3],
                "first_event_stage_1": metrics["first_event_counts"][0],
                "first_event_stage_2": metrics["first_event_counts"][1],
                "first_event_stage_3": metrics["first_event_counts"][2],
                "first_event_stage_4": metrics["first_event_counts"][3],
            }
        )
    return pd.DataFrame(rows)


def paired_onset_tests(sample_df: pd.DataFrame, onset_col: str, event_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        sample_df[["prompt_id", "model", onset_col]]
        .pivot(index="prompt_id", columns="model", values=onset_col)
        .reindex(columns=ordered_models(sample_df["model"].astype(str).unique()))
    )

    friedman_row = {
        "event_type": event_name,
        "test": "friedman",
        "n_prompts": len(pivot),
        "statistic": math.nan,
        "p_value": math.nan,
    }
    valid = pivot.dropna()
    if valid.shape[0] > 0 and valid.shape[1] >= 3:
        stat, p_value = friedmanchisquare(*[valid[col].values for col in valid.columns])
        friedman_row["statistic"] = float(stat)
        friedman_row["p_value"] = float(p_value)

    pairwise_rows: List[Dict] = []
    cols = list(valid.columns)
    raw_p_values: List[float] = []
    pending_indexes: List[int] = []
    for idx_a in range(len(cols)):
        for idx_b in range(idx_a + 1, len(cols)):
            model_a = cols[idx_a]
            model_b = cols[idx_b]
            diff = valid[model_a] - valid[model_b]
            row = {
                "event_type": event_name,
                "model_a": model_a,
                "model_b": model_b,
                "n_prompts": int(len(valid)),
                "median_diff_a_minus_b": float(np.median(diff)),
                "mean_diff_a_minus_b": float(np.mean(diff)),
                "statistic": math.nan,
                "p_value_raw": math.nan,
                "p_value_holm": math.nan,
            }
            try:
                stat, p_value = wilcoxon(valid[model_a], valid[model_b], zero_method="pratt", alternative="two-sided")
                row["statistic"] = float(stat)
                row["p_value_raw"] = float(p_value)
                raw_p_values.append(float(p_value))
                pending_indexes.append(len(pairwise_rows))
            except ValueError:
                pass
            pairwise_rows.append(row)

    if raw_p_values:
        _, corrected, _, _ = multipletests(raw_p_values, method="holm")
        for row_index, corrected_p in zip(pending_indexes, corrected):
            pairwise_rows[row_index]["p_value_holm"] = float(corrected_p)

    return pd.DataFrame([friedman_row]), pd.DataFrame(pairwise_rows)


def stage_binary_tests(sample_df: pd.DataFrame, onset_col: str, event_name: str) -> pd.DataFrame:
    models = ordered_models(sample_df["model"].astype(str).unique())
    rows: List[Dict] = []
    for stage_idx in range(1, 5):
        subset = sample_df.copy()
        subset[f"event_by_stage_{stage_idx}"] = subset[onset_col].apply(
            lambda value: int(not pd.isna(value) and int(value) <= stage_idx)
        )
        matrix = (
            subset[["prompt_id", "model", f"event_by_stage_{stage_idx}"]]
            .pivot(index="prompt_id", columns="model", values=f"event_by_stage_{stage_idx}")
            .reindex(columns=models)
        )
        valid = matrix.dropna()
        statistic = math.nan
        p_value = math.nan
        if valid.shape[0] > 0 and valid.shape[1] >= 3:
            test_result = cochrans_q(valid.values)
            statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)
        rows.append(
            {
                "event_type": event_name,
                "stage_index": stage_idx,
                "stage_label": STAGE_LABELS[STAGE_ORDER[stage_idx - 1]],
                "n_prompts": int(valid.shape[0]),
                "cochrans_q": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def collapse_pattern_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        sample_df.groupby(["model", "collapse_pattern"], observed=True)
        .size()
        .reset_index(name="count")
    )
    total = summary.groupby("model", observed=True)["count"].transform("sum")
    summary["proportion"] = summary["count"] / total
    summary["model_label"] = summary["model"].astype(str).map(display_model)
    return summary


def fit_gee(turn_df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    formula = (
        f"{outcome_col} ~ C(model) + stage_index + stage_alignment + "
        "stage_leakage + mechanism_gain + num_selected_dimensions"
    )
    needed = [
        outcome_col,
        "model",
        "sample_id",
        "stage_index",
        "stage_alignment",
        "stage_leakage",
        "mechanism_gain",
        "num_selected_dimensions",
    ]
    df = turn_df.dropna(subset=needed).copy()
    if df.empty:
        return pd.DataFrame()

    model = GEE.from_formula(
        formula=formula,
        groups="sample_id",
        cov_struct=Exchangeable(),
        family=Binomial(),
        data=df,
    )
    result = model.fit()
    if not np.all(np.isfinite(result.params.values)):
        return pd.DataFrame(
            [
                {
                    "outcome": outcome_col,
                    "term": "model_unstable",
                    "coef_log_odds": math.nan,
                    "odds_ratio": math.nan,
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "p_value": math.nan,
                }
            ]
        )
    conf = result.conf_int()

    rows: List[Dict] = []
    for name, coef, p_value in zip(result.params.index, result.params.values, result.pvalues.values):
        lower = conf.loc[name, 0]
        upper = conf.loc[name, 1]
        rows.append(
            {
                "outcome": outcome_col,
                "term": name,
                "coef_log_odds": float(coef),
                "odds_ratio": float(math.exp(coef)),
                "ci_low": float(math.exp(lower)),
                "ci_high": float(math.exp(upper)),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def turn_level_correlations(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for predictor in ["stage_index", "stage_alignment", "stage_leakage", "mechanism_gain", "num_selected_dimensions"]:
        subset = turn_df[["score", predictor]].dropna()
        if subset.empty:
            continue
        rho, rho_p = spearmanr(subset["score"], subset[predictor])
        tau, tau_p = kendalltau(subset["score"], subset[predictor])
        rows.append(
            {
                "predictor": predictor,
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
                "n": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def model_stage_score_summary(turn_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        turn_df.groupby(["model", "stage_index", "stage_label"], observed=True)["score"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
    grouped["model_label"] = grouped["model"].astype(str).map(display_model)
    return grouped


def stage_predictor_summary(turn_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        turn_df.groupby(["model", "stage_index", "stage_label"], observed=True)[
            ["score", "stage_alignment", "stage_leakage", "mechanism_gain", "num_selected_dimensions"]
        ]
        .mean()
        .reset_index()
    )
    summary["model_label"] = summary["model"].astype(str).map(display_model)
    return summary


def dimension_activation_summary(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    grouped = turn_df.groupby(["model", "stage_index", "stage_label"], observed=True)
    for (model, stage_index, stage_label), subset in grouped:
        for dimension in ALL_DIMENSIONS:
            column = dim_col(dimension)
            rows.append(
                {
                    "model": str(model),
                    "model_label": display_model(str(model)),
                    "stage_index": int(stage_index),
                    "stage_label": stage_label,
                    "dimension": dimension,
                    "activation_rate": float(subset[column].mean()),
                    "activation_count": int(subset[column].sum()),
                    "turn_count": int(len(subset)),
                    "mean_score_when_active": float(subset.loc[subset[column] == 1, "score"].mean())
                    if int(subset[column].sum()) > 0
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def dimension_score_associations(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    p_values: List[float] = []
    row_indexes: List[int] = []
    for dimension in ALL_DIMENSIONS:
        column = dim_col(dimension)
        subset = turn_df[["score", column]].dropna()
        if subset.empty:
            continue
        rho, rho_p = spearmanr(subset["score"], subset[column])
        tau, tau_p = kendalltau(subset["score"], subset[column])
        active_scores = subset.loc[subset[column] == 1, "score"]
        inactive_scores = subset.loc[subset[column] == 0, "score"]
        row = {
            "dimension": dimension,
            "n_turns": int(len(subset)),
            "activation_rate": float(subset[column].mean()),
            "mean_score_when_active": float(active_scores.mean()) if len(active_scores) else math.nan,
            "mean_score_when_inactive": float(inactive_scores.mean()) if len(inactive_scores) else math.nan,
            "mean_diff_active_minus_inactive": (
                float(active_scores.mean() - inactive_scores.mean())
                if len(active_scores) and len(inactive_scores)
                else math.nan
            ),
            "spearman_rho": float(rho),
            "spearman_p_raw": float(rho_p),
            "spearman_p_holm": math.nan,
            "kendall_tau": float(tau),
            "kendall_p_raw": float(tau_p),
            "kendall_p_holm": math.nan,
        }
        rows.append(row)
        p_values.append(float(rho_p))
        row_indexes.append(len(rows) - 1)

    if p_values:
        _, corrected, _, _ = multipletests(p_values, method="holm")
        for row_index, corrected_p in zip(row_indexes, corrected):
            rows[row_index]["spearman_p_holm"] = float(corrected_p)
            rows[row_index]["kendall_p_holm"] = float(corrected_p)
    return pd.DataFrame(rows)


def fit_dimension_gee(turn_df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    rows: List[Dict] = []
    raw_p_values: List[float] = []
    pending_indexes: List[int] = []
    for dimension in ALL_DIMENSIONS:
        column = dim_col(dimension)
        needed = [
            outcome_col,
            "model",
            "sample_id",
            "stage_index",
            "stage_alignment",
            "stage_leakage",
            column,
        ]
        df = turn_df.dropna(subset=needed).copy()
        if df.empty or df[column].nunique() < 2:
            continue
        try:
            model = GEE.from_formula(
                formula=(
                    f"{outcome_col} ~ C(model) + stage_index + "
                    f"stage_alignment + stage_leakage + {column}"
                ),
                groups="sample_id",
                cov_struct=Exchangeable(),
                family=Binomial(),
                data=df,
            )
            result = model.fit()
            if not np.all(np.isfinite(result.params.values)):
                continue
            conf = result.conf_int()
            coef = result.params[column]
            p_value = result.pvalues[column]
            lower = conf.loc[column, 0]
            upper = conf.loc[column, 1]
            rows.append(
                {
                    "outcome": outcome_col,
                    "dimension": dimension,
                    "activation_rate": float(df[column].mean()),
                    "coef_log_odds": float(coef),
                    "odds_ratio": float(math.exp(coef)),
                    "ci_low": float(math.exp(lower)),
                    "ci_high": float(math.exp(upper)),
                    "p_value_raw": float(p_value),
                    "p_value_holm": math.nan,
                }
            )
            raw_p_values.append(float(p_value))
            pending_indexes.append(len(rows) - 1)
        except Exception:
            continue

    if raw_p_values:
        _, corrected, _, _ = multipletests(raw_p_values, method="holm")
        for row_index, corrected_p in zip(pending_indexes, corrected):
            rows[row_index]["p_value_holm"] = float(corrected_p)
    return pd.DataFrame(rows)


def maybe_run_reasoning_proxy(
    reasoning_proxy_csv: Path,
    sample_df: pd.DataFrame,
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    if not reasoning_proxy_csv.exists():
        return None

    proxy_df = pd.read_csv(reasoning_proxy_csv)
    if "model" not in proxy_df.columns or "reasoning_proxy" not in proxy_df.columns:
        return None

    proxy_df["model"] = proxy_df["model"].astype(str).map(normalize_model_name)
    proxy_df["reasoning_proxy"] = pd.to_numeric(proxy_df["reasoning_proxy"], errors="coerce")
    proxy_df = proxy_df.dropna(subset=["reasoning_proxy"])
    if proxy_df.empty:
        return None

    soft_metrics = summarize_event_metrics(sample_df, "first_softening_stage_idx", "softening")
    comp_metrics = summarize_event_metrics(sample_df, "first_compromise_stage_idx", "compromise")
    merged = proxy_df.merge(
        soft_metrics[["model", "C1", "AUC_C", "RMST"]].rename(
            columns={"C1": "soft_C1", "AUC_C": "soft_AUC_C", "RMST": "soft_RMST"}
        ),
        on="model",
        how="inner",
    ).merge(
        comp_metrics[["model", "C1", "AUC_C", "RMST"]].rename(
            columns={"C1": "comp_C1", "AUC_C": "comp_AUC_C", "RMST": "comp_RMST"}
        ),
        on="model",
        how="inner",
    )
    if len(merged) < 3:
        return None

    rows: List[Dict] = []
    for metric in ["soft_C1", "soft_AUC_C", "soft_RMST", "comp_C1", "comp_AUC_C", "comp_RMST"]:
        rho, rho_p = spearmanr(merged["reasoning_proxy"], merged[metric])
        tau, tau_p = kendalltau(merged["reasoning_proxy"], merged[metric])
        rows.append(
            {
                "metric": metric,
                "n_models": int(len(merged)),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "kendall_tau": float(tau),
                "kendall_p": float(tau_p),
            }
        )
    result_df = pd.DataFrame(rows)
    merged.to_csv(output_dir / "reasoning_proxy_merged.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(output_dir / "reasoning_proxy_correlations.csv", index=False, encoding="utf-8-sig")
    return result_df


def plot_cumulative_curves(metrics_df: pd.DataFrame, output_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    panels = [
        ("softening", axes[0], "Softening: first score >= 3"),
        ("compromise", axes[1], "Compromise: first successful turn"),
    ]
    stages = [1, 2, 3, 4]
    for event_type, ax, subtitle in panels:
        subset = metrics_df[metrics_df["event_type"] == event_type].copy()
        for _, row in subset.iterrows():
            model = str(row["model"])
            ax.plot(
                stages,
                [row["C1"], row["C2"], row["C3"], row["C4"]],
                marker="o",
                linewidth=2.3,
                color=MODEL_COLORS.get(model),
                label=display_model(model),
            )
        ax.set_xticks(stages)
        ax.set_xticklabels([f"S{stage}" for stage in stages])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Cumulative proportion")
        ax.set_title(subtitle)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hazard_heatmap(metrics_df: pd.DataFrame, event_type: str, output_path: Path, title: str) -> None:
    subset = metrics_df[metrics_df["event_type"] == event_type].copy()
    subset = subset.set_index("model").reindex(ordered_models(subset["model"].tolist())).reset_index()
    heatmap_df = subset.set_index("model")[["h1", "h2", "h3", "h4"]]
    heatmap_df.index = [display_model(idx) for idx in heatmap_df.index]
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(heatmap_df, annot=True, cmap="YlOrRd", vmin=0, vmax=1, fmt=".2f", cbar_kws={"label": "Hazard"})
    plt.title(title)
    plt.xlabel("Stage")
    plt.ylabel("Model")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_scores(score_summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for model in ordered_models(score_summary_df["model"].astype(str).unique()):
        subset = score_summary_df[score_summary_df["model"].astype(str) == model].sort_values("stage_index")
        ax.plot(
            subset["stage_index"],
            subset["mean"],
            marker="o",
            linewidth=2.3,
            color=MODEL_COLORS.get(model),
            label=display_model(model),
        )
        ax.fill_between(
            subset["stage_index"],
            subset["mean"] - subset["se"],
            subset["mean"] + subset["se"],
            color=MODEL_COLORS.get(model),
            alpha=0.15,
        )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax.set_ylim(1, 5.05)
    ax.set_xlabel("Executed stage")
    ax.set_ylabel("Mean judgement score")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stage_distribution(sample_df: pd.DataFrame, onset_col: str, output_path: Path, title: str) -> None:
    stages = [1, 2, 3, 4]
    rows = []
    models = ordered_models(sample_df["model"].astype(str).unique())
    for model in models:
        subset = sample_df[sample_df["model"].astype(str) == model]
        counts = Counter(
            int(value)
            for value in subset[onset_col].dropna().astype(int).tolist()
            if 1 <= int(value) <= 4
        )
        total = len(subset)
        for stage in stages:
            rows.append(
                {
                    "model": model,
                    "stage_index": stage,
                    "proportion": counts.get(stage, 0) / total if total else 0.0,
                }
            )
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(9.5, 5.6))
    bottom = np.zeros(len(models))
    x = np.arange(len(models))
    palette = ["#d9e6f5", "#a8c6ea", "#6e9dd2", "#315f9b"]
    for idx, stage in enumerate(stages):
        values = [
            plot_df[(plot_df["model"] == model) & (plot_df["stage_index"] == stage)]["proportion"].iloc[0]
            for model in models
        ]
        plt.bar(x, values, bottom=bottom, color=palette[idx], label=f"S{stage}")
        bottom += np.array(values)
    plt.xticks(x, [display_model(model) for model in models], rotation=20, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("Proportion of prompts")
    plt.title(title)
    plt.legend(frameon=False, title="First onset stage")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_collapse_patterns(collapse_df: pd.DataFrame, output_path: Path, title: str) -> None:
    models = ordered_models(collapse_df["model"].astype(str).unique())
    pivot = (
        collapse_df.pivot(index="model", columns="collapse_pattern", values="proportion")
        .reindex(models)
        .fillna(0.0)
    )
    pivot.index = [display_model(model) for model in pivot.index]
    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8.5, 5.0),
        color={"abrupt": "#df6b57", "gradual": "#5aa469", "not_compromised": "#9aa0a6"},
    )
    ax.set_ylabel("Proportion of prompts")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, title="Pattern")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prompt_onset_heatmap(sample_df: pd.DataFrame, onset_col: str, output_path: Path, title: str) -> None:
    models = ordered_models(sample_df["model"].astype(str).unique())
    heatmap = (
        sample_df[["prompt_id", "model", onset_col]]
        .pivot(index="prompt_id", columns="model", values=onset_col)
        .reindex(columns=models)
    )
    heatmap.columns = [display_model(col) for col in heatmap.columns]
    plt.figure(figsize=(8.5, 7.0))
    sns.heatmap(heatmap, annot=True, cmap="Blues", vmin=1, vmax=4, fmt=".0f", cbar_kws={"label": "Stage"})
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Prompt")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dimension_stage_heatmap(dimension_summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    aggregated = (
        dimension_summary_df.groupby(["stage_index", "dimension"], observed=True)["activation_rate"]
        .mean()
        .reset_index()
    )
    heatmap = aggregated.pivot(index="dimension", columns="stage_index", values="activation_rate")
    heatmap = heatmap.reindex(index=ALL_DIMENSIONS, columns=[1, 2, 3, 4])
    plt.figure(figsize=(8.5, 7.5))
    sns.heatmap(heatmap, annot=True, cmap="Greens", vmin=0, vmax=1, fmt=".2f", cbar_kws={"label": "Activation rate"})
    plt.title(title)
    plt.xlabel("Stage")
    plt.ylabel("Dimension")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dimension_forest(gee_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if gee_df.empty:
        return
    plot_df = gee_df.sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8.8, max(5.6, 0.32 * len(plot_df) + 1.8)))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["odds_ratio"],
        y,
        xerr=[
            plot_df["odds_ratio"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["odds_ratio"],
        ],
        fmt="o",
        color="#355c7d",
        ecolor="#99b2c6",
        elinewidth=2,
        capsize=3,
    )
    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["dimension"])
    ax.set_xlabel("Odds ratio")
    ax.set_title(title)
    ax.set_xscale("log")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results_snapshot(
    sample_df: pd.DataFrame,
    event_metrics_df: pd.DataFrame,
    onset_tests_df: pd.DataFrame,
    stage_tests_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    turn_corr_df: pd.DataFrame,
    dimension_score_df: pd.DataFrame,
    dimension_activation_df: pd.DataFrame,
    gee_soft_df: pd.DataFrame,
    gee_strong_df: pd.DataFrame,
    gee_comp_df: pd.DataFrame,
    gee_dimension_soft_df: pd.DataFrame,
    gee_dimension_strong_df: pd.DataFrame,
    reasoning_corr_df: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    summary = {
        "dataset": {
            "models": [display_model(model) for model in ordered_models(sample_df["model"].astype(str).unique())],
            "n_models": int(sample_df["model"].nunique()),
            "n_prompts": int(sample_df["prompt_id"].nunique()),
            "n_samples": int(len(sample_df)),
        },
        "event_metrics": event_metrics_df.to_dict(orient="records"),
        "onset_tests": onset_tests_df.to_dict(orient="records"),
        "stage_tests": stage_tests_df.to_dict(orient="records"),
        "collapse_patterns": collapse_df.to_dict(orient="records"),
        "turn_level_correlations": turn_corr_df.to_dict(orient="records"),
        "dimension_score_associations": dimension_score_df.to_dict(orient="records"),
        "dimension_activation_summary": dimension_activation_df.to_dict(orient="records"),
        "gee_softened": gee_soft_df.to_dict(orient="records"),
        "gee_strong_compliance": gee_strong_df.to_dict(orient="records"),
        "gee_compromised": gee_comp_df.to_dict(orient="records"),
        "gee_dimensions_softened": gee_dimension_soft_df.to_dict(orient="records"),
        "gee_dimensions_strong_compliance": gee_dimension_strong_df.to_dict(orient="records"),
        "reasoning_proxy_correlations": None if reasoning_corr_df is None else reasoning_corr_df.to_dict(orient="records"),
    }
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    summary_paths = [Path(path) for path in args.summary]
    sample_df, turn_df = build_long_tables(summary_paths)

    sample_df.to_csv(output_dir / "sample_level_long.csv", index=False, encoding="utf-8-sig")
    turn_df.to_csv(output_dir / "turn_level_analysis.csv", index=False, encoding="utf-8-sig")

    event_metrics_soft = summarize_event_metrics(sample_df, "first_softening_stage_idx", "softening")
    event_metrics_comp = summarize_event_metrics(sample_df, "first_compromise_stage_idx", "compromise")
    event_metrics_df = pd.concat([event_metrics_soft, event_metrics_comp], ignore_index=True)
    event_metrics_df.to_csv(output_dir / "event_metrics_by_model.csv", index=False, encoding="utf-8-sig")

    friedman_soft_df, pairwise_soft_df = paired_onset_tests(sample_df, "first_softening_stage_idx", "softening")
    friedman_comp_df, pairwise_comp_df = paired_onset_tests(sample_df, "first_compromise_stage_idx", "compromise")
    onset_tests_df = pd.concat([friedman_soft_df, friedman_comp_df], ignore_index=True)
    onset_tests_df.to_csv(output_dir / "onset_friedman_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat([pairwise_soft_df, pairwise_comp_df], ignore_index=True).to_csv(
        output_dir / "onset_pairwise_wilcoxon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stage_soft_df = stage_binary_tests(sample_df, "first_softening_stage_idx", "softening")
    stage_comp_df = stage_binary_tests(sample_df, "first_compromise_stage_idx", "compromise")
    stage_tests_df = pd.concat([stage_soft_df, stage_comp_df], ignore_index=True)
    stage_tests_df.to_csv(output_dir / "stage_cochrans_q_tests.csv", index=False, encoding="utf-8-sig")

    collapse_df = collapse_pattern_summary(sample_df)
    collapse_df.to_csv(output_dir / "collapse_pattern_summary.csv", index=False, encoding="utf-8-sig")

    turn_corr_df = turn_level_correlations(turn_df)
    turn_corr_df.to_csv(output_dir / "turn_level_correlations.csv", index=False, encoding="utf-8-sig")

    dimension_activation_df = dimension_activation_summary(turn_df)
    dimension_activation_df.to_csv(output_dir / "dimension_activation_summary.csv", index=False, encoding="utf-8-sig")
    dimension_score_df = dimension_score_associations(turn_df)
    dimension_score_df.to_csv(output_dir / "dimension_score_associations.csv", index=False, encoding="utf-8-sig")

    gee_soft_df = fit_gee(turn_df, "softened")
    gee_strong_df = fit_gee(turn_df, "strong_compliance")
    gee_comp_df = fit_gee(turn_df, "compromised")
    gee_dimension_soft_df = fit_dimension_gee(turn_df, "softened")
    gee_dimension_strong_df = fit_dimension_gee(turn_df, "strong_compliance")
    gee_soft_df.to_csv(output_dir / "gee_softened.csv", index=False, encoding="utf-8-sig")
    gee_strong_df.to_csv(output_dir / "gee_strong_compliance.csv", index=False, encoding="utf-8-sig")
    gee_comp_df.to_csv(output_dir / "gee_compromised.csv", index=False, encoding="utf-8-sig")
    gee_dimension_soft_df.to_csv(output_dir / "gee_dimensions_softened.csv", index=False, encoding="utf-8-sig")
    gee_dimension_strong_df.to_csv(output_dir / "gee_dimensions_strong_compliance.csv", index=False, encoding="utf-8-sig")

    score_summary_df = model_stage_score_summary(turn_df)
    score_summary_df.to_csv(output_dir / "model_stage_score_summary.csv", index=False, encoding="utf-8-sig")
    predictor_summary_df = stage_predictor_summary(turn_df)
    predictor_summary_df.to_csv(output_dir / "stage_predictor_summary.csv", index=False, encoding="utf-8-sig")

    reasoning_proxy_path = Path(args.reasoning_proxy_csv)
    write_reasoning_proxy_template(reasoning_proxy_path, sample_df["model"].astype(str).unique())
    reasoning_corr_df = maybe_run_reasoning_proxy(reasoning_proxy_path, sample_df, output_dir)

    plot_cumulative_curves(
        event_metrics_df,
        figures_dir / "cumulative_curves_softening_compromise.png",
        "Cumulative Vulnerability Curves Across Models",
    )
    plot_hazard_heatmap(
        event_metrics_df,
        "softening",
        figures_dir / "hazard_heatmap_softening.png",
        "Stage-Specific Softening Hazard",
    )
    plot_hazard_heatmap(
        event_metrics_df,
        "compromise",
        figures_dir / "hazard_heatmap_compromise.png",
        "Stage-Specific Compromise Hazard",
    )
    plot_mean_scores(
        score_summary_df,
        figures_dir / "mean_score_by_stage.png",
        "Executed-Turn Mean Judgement Score by Stage",
    )
    plot_stage_distribution(
        sample_df,
        "first_softening_stage_idx",
        figures_dir / "first_softening_stage_distribution.png",
        "First Softening Stage Distribution",
    )
    plot_stage_distribution(
        sample_df,
        "first_compromise_stage_idx",
        figures_dir / "first_compromise_stage_distribution.png",
        "First Compromise Stage Distribution",
    )
    plot_collapse_patterns(
        collapse_df,
        figures_dir / "collapse_patterns.png",
        "Gradual vs Abrupt Collapse After Softening",
    )
    plot_prompt_onset_heatmap(
        sample_df,
        "first_softening_stage_idx",
        figures_dir / "prompt_heatmap_first_softening.png",
        "Prompt-Matched First Softening Stage",
    )
    plot_prompt_onset_heatmap(
        sample_df,
        "first_compromise_stage_idx",
        figures_dir / "prompt_heatmap_first_compromise.png",
        "Prompt-Matched First Compromise Stage",
    )
    plot_dimension_stage_heatmap(
        dimension_activation_df,
        figures_dir / "dimension_stage_activation_heatmap.png",
        "Average Dimension Activation Rate by Stage",
    )
    plot_dimension_forest(
        gee_dimension_soft_df,
        figures_dir / "dimension_forest_softened.png",
        "Dimension-Level ORs for Softening",
    )
    plot_dimension_forest(
        gee_dimension_strong_df,
        figures_dir / "dimension_forest_strong_compliance.png",
        "Dimension-Level ORs for Strong Compliance",
    )

    write_results_snapshot(
        sample_df,
        event_metrics_df,
        onset_tests_df,
        stage_tests_df,
        collapse_df,
        turn_corr_df,
        dimension_score_df,
        dimension_activation_df,
        gee_soft_df,
        gee_strong_df,
        gee_comp_df,
        gee_dimension_soft_df,
        gee_dimension_strong_df,
        reasoning_corr_df,
        output_dir / "results_snapshot.json",
    )

    print(f"Generated analysis outputs in: {output_dir}")


if __name__ == "__main__":
    main()
