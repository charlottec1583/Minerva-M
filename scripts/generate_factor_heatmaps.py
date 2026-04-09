from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_STAGE_ORDER = [
    "context_establishment",
    "relationship_building",
    "constraint_induction",
    "escalation",
]

STAGE_DISPLAY_NAMES = {
    "context_establishment": "Context",
    "relationship_building": "Relationship",
    "constraint_induction": "Constraint",
    "escalation": "Escalation",
}

STAGE_ALLOWED_FACTORS = {
    "context_establishment": {
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
        "context_plausibility",
    },
    "relationship_building": {
        "affinity",
        "warmth",
        "self_efficacy",
        "value_alignment",
        "context_plausibility",
        "operational_capability",
        "task_representation_clarity",
        "scope_framing",
        "gain",
        "cognitive_dissonance",
    },
    "constraint_induction": {
        "operational_capability",
        "task_representation_clarity",
        "scope_framing",
        "norm_approval",
        "urgency",
        "context_plausibility",
        "self_efficacy",
        "legitimacy",
        "authority",
        "value_alignment",
        "cognitive_dissonance",
        "commitment_consistency_pressure",
    },
    "escalation": {
        "gain",
        "loss",
        "cognitive_dissonance",
        "commitment_consistency_pressure",
        "urgency",
        "context_plausibility",
        "self_efficacy",
        "operational_capability",
        "task_representation_clarity",
        "legitimacy",
        "authority",
        "norm_approval",
        "value_alignment",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stage x factor heatmaps from an analysis_long_table CSV. "
            "Outputs both success-turn usage and uplift heatmaps as SVG plus CSV tables."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to analysis_long_table_*.csv",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Target model name to analyze. If omitted and only one model exists, it is auto-selected.",
    )
    parser.add_argument(
        "--cell-id",
        default=None,
        help="Optional cell_id filter. If omitted, aggregate across all cell_ids for the selected model.",
    )
    parser.add_argument(
        "--output-dir",
        default="presentation_charts",
        help="Directory for generated SVG/CSV outputs. Defaults to presentation_charts",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def parse_boolish(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    return 0


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def read_round_factor_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if str(row.get("row_type", "")).strip() == "round_factor"]


def resolve_target_model(rows: Sequence[Dict[str, str]], requested: Optional[str]) -> str:
    models = []
    seen = set()
    for row in rows:
        model = str(row.get("target_model", "")).strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)

    if requested:
        if requested not in seen:
            raise ValueError(
                f"target model `{requested}` not found in CSV. Available models: {', '.join(models)}"
            )
        return requested

    if len(models) == 1:
        return models[0]

    raise ValueError(
        "Multiple target_model values found. Please pass --target-model. "
        f"Available models: {', '.join(models)}"
    )


def filter_rows(
    rows: Sequence[Dict[str, str]],
    target_model: str,
    cell_id: Optional[str],
) -> List[Dict[str, str]]:
    filtered = []
    for row in rows:
        if str(row.get("target_model", "")).strip() != target_model:
            continue
        if cell_id is not None and str(row.get("cell_id", "")).strip() != cell_id:
            continue
        filtered.append(row)
    return filtered


def infer_stage_order(stages_seen: Iterable[str]) -> List[str]:
    seen_list = list(dict.fromkeys(stage for stage in stages_seen if stage))
    ordered = [stage for stage in DEFAULT_STAGE_ORDER if stage in seen_list]
    ordered.extend(stage for stage in seen_list if stage not in ordered)
    return ordered


def infer_factor_order(rows: Sequence[Dict[str, str]]) -> List[str]:
    factors: List[str] = []
    seen = set()
    for row in rows:
        factor = str(row.get("factor", "")).strip()
        if factor and factor not in seen:
            seen.add(factor)
            factors.append(factor)
    return factors


def is_factor_allowed(stage: str, factor: str) -> bool:
    allowed = STAGE_ALLOWED_FACTORS.get(stage)
    if allowed is None:
        return True
    return factor in allowed


def make_turn_id(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        str(row.get("target_model", "")).strip(),
        str(row.get("cell_id", "")).strip(),
        str(row.get("index", "")).strip(),
        str(row.get("sample_index", "")).strip(),
        str(row.get("stage_name", "")).strip(),
        str(row.get("iteration", "")).strip(),
    )


def compute_stage_counts(
    rows: Sequence[Dict[str, str]],
    stage_order: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    total_turns = defaultdict(set)
    success_turns = defaultdict(set)

    for row in rows:
        stage = str(row.get("stage_name", "")).strip()
        if not stage:
            continue
        turn_id = make_turn_id(row)
        total_turns[stage].add(turn_id)
        if parse_boolish(row.get("is_success_round", 0)):
            success_turns[stage].add(turn_id)

    result = {}
    for stage in stage_order:
        result[stage] = {
            "total_turns": len(total_turns.get(stage, set())),
            "success_turns": len(success_turns.get(stage, set())),
        }
    return result


def compute_stage_factor_rates(
    rows: Sequence[Dict[str, str]],
    stage_order: Sequence[str],
    factor_order: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, Optional[float]]],
    Dict[str, Dict[str, Optional[float]]],
    Dict[str, Dict[str, Optional[float]]],
    Dict[str, Dict[str, Optional[float]]],
]:
    all_sum = defaultdict(float)
    all_count = defaultdict(int)
    success_sum = defaultdict(float)
    success_count = defaultdict(int)
    allowed_sum = defaultdict(float)
    allowed_count = defaultdict(int)
    global_allowed_sum = defaultdict(float)
    global_allowed_count = defaultdict(int)

    for row in rows:
        stage = str(row.get("stage_name", "")).strip()
        factor = str(row.get("factor", "")).strip()
        if not stage or not factor:
            continue

        key = (stage, factor)
        has_factor = float(parse_boolish(row.get("has_factor", 0)))
        all_sum[key] += has_factor
        all_count[key] += 1
        if is_factor_allowed(stage, factor):
            allowed_sum[key] += has_factor
            allowed_count[key] += 1
            global_allowed_sum[factor] += has_factor
            global_allowed_count[factor] += 1

        if parse_boolish(row.get("is_success_round", 0)):
            success_sum[key] += has_factor
            success_count[key] += 1

    success_usage: Dict[str, Dict[str, Optional[float]]] = {}
    baseline_usage: Dict[str, Dict[str, Optional[float]]] = {}
    global_allowed_baseline: Dict[str, Dict[str, Optional[float]]] = {}
    uplift: Dict[str, Dict[str, Optional[float]]] = {}

    for stage in stage_order:
        success_usage[stage] = {}
        baseline_usage[stage] = {}
        global_allowed_baseline[stage] = {}
        uplift[stage] = {}
        for factor in factor_order:
            key = (stage, factor)
            allowed = is_factor_allowed(stage, factor)
            if allowed and allowed_count[key]:
                baseline = allowed_sum[key] / allowed_count[key]
                baseline_usage[stage][factor] = baseline
            elif allowed:
                baseline = None
                baseline_usage[stage][factor] = None
            else:
                baseline = None
                baseline_usage[stage][factor] = None

            if global_allowed_count[factor]:
                global_allowed_baseline[stage][factor] = (
                    global_allowed_sum[factor] / global_allowed_count[factor]
                )
            else:
                global_allowed_baseline[stage][factor] = None

            if success_count[key]:
                success_val = success_sum[key] / success_count[key]
                success_usage[stage][factor] = success_val
                uplift[stage][factor] = (
                    success_val - baseline if baseline is not None else None
                )
            else:
                success_usage[stage][factor] = None
                uplift[stage][factor] = None

    return success_usage, baseline_usage, global_allowed_baseline, uplift


def save_matrix_csv(
    output_path: Path,
    stage_order: Sequence[str],
    factor_order: Sequence[str],
    matrix: Dict[str, Dict[str, Optional[float]]],
    value_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage_name", *factor_order])
        for stage in stage_order:
            row = [stage]
            for factor in factor_order:
                value = matrix.get(stage, {}).get(factor)
                if value is None:
                    row.append("")
                else:
                    row.append(f"{value:.6f}")
            writer.writerow(row)
    print(f"Generated: {output_path} ({value_label})")


def save_stage_counts_csv(
    output_path: Path,
    stage_order: Sequence[str],
    stage_counts: Dict[str, Dict[str, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage_name", "success_turns", "total_turns"])
        for stage in stage_order:
            counts = stage_counts.get(stage, {})
            writer.writerow([stage, counts.get("success_turns", 0), counts.get("total_turns", 0)])
    print(f"Generated: {output_path} (stage counts)")


def svg_wrap(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
<rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="20" ry="20" fill="#ffffff" stroke="#e5e7eb" stroke-width="1.2"/>
<style>
  text {{ font-family: "Aptos", "Segoe UI", Arial, sans-serif; fill: #1f2937; }}
  .title {{ font-size: 28px; font-weight: 700; }}
  .subtitle {{ font-size: 15px; fill: #475569; }}
  .axis-label {{ font-size: 15px; font-weight: 600; fill: #111827; }}
  .tick {{ font-size: 13px; fill: #334155; }}
  .cell-label {{ font-size: 12px; font-weight: 600; }}
  .legend-title {{ font-size: 14px; font-weight: 700; }}
  .legend-text {{ font-size: 12px; fill: #475569; }}
</style>
{body}
</svg>
"""


def wrap_factor_label(factor: str, max_line_chars: int = 14) -> List[str]:
    words = factor.replace("_", " ").split()
    if not words:
        return [factor]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_line_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def blend_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = clamp(ratio, 0.0, 1.0)
    a = tuple(int(color_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(x + (y - x) * ratio) for x, y in zip(a, b))
    return "#" + "".join(f"{v:02x}" for v in mixed)


def usage_color(value: Optional[float]) -> str:
    if value is None:
        return "#f1f5f9"
    return blend_hex("#f8fbff", "#1d4ed8", clamp(value, 0.0, 1.0))


def uplift_color(value: Optional[float], max_abs: float) -> str:
    if value is None:
        return "#f1f5f9"
    if max_abs <= 0:
        return "#f8fafc"
    norm = clamp(value / max_abs, -1.0, 1.0)
    if norm >= 0:
        return blend_hex("#fff7ed", "#c2410c", norm)
    return blend_hex("#eff6ff", "#1d4ed8", abs(norm))


def ideal_text_color(fill_hex: str) -> str:
    r, g, b = (int(fill_hex[i : i + 2], 16) for i in (1, 3, 5))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 145 else "#111827"


def draw_gradient_legend(
    body: List[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    labels: Sequence[Tuple[float, str]],
    gradient_id: str,
    stops: Sequence[Tuple[str, float]],
) -> None:
    body.append(f'<defs><linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="0%">')
    for color, offset in stops:
        body.append(f'<stop offset="{offset:.1f}%" stop-color="{color}"/>')
    body.append("</linearGradient></defs>")
    body.append(f'<text class="legend-title" x="{x}" y="{y - 10:.1f}">{title}</text>')
    body.append(
        f'<rect x="{x}" y="{y:.1f}" width="{width}" height="{height}" '
        f'fill="url(#{gradient_id})" stroke="#d1d5db" stroke-width="1"/>'
    )
    for pos, label in labels:
        tx = x + pos * width
        body.append(f'<line x1="{tx:.1f}" y1="{y + height:.1f}" x2="{tx:.1f}" y2="{y + height + 6:.1f}" stroke="#6b7280" stroke-width="1"/>')
        body.append(f'<text class="legend-text" x="{tx:.1f}" y="{y + height + 20:.1f}" text-anchor="middle">{label}</text>')


def build_heatmap_svg(
    title: str,
    subtitle_lines: Sequence[str],
    stage_order: Sequence[str],
    factor_order: Sequence[str],
    matrix: Dict[str, Dict[str, Optional[float]]],
    stage_counts: Dict[str, Dict[str, int]],
    mode: str,
) -> str:
    cell_w = 170
    cell_h = 42
    left = 280
    right = 90
    top = 170
    bottom = 120
    width = left + right + cell_w * len(stage_order)
    height = top + bottom + cell_h * len(factor_order)

    body: List[str] = []
    body.append(f'<text class="title" x="{width / 2:.1f}" y="58" text-anchor="middle">{title}</text>')
    for idx, line in enumerate(subtitle_lines, 1):
        body.append(
            f'<text class="subtitle" x="{width / 2:.1f}" y="{58 + idx * 22:.1f}" text-anchor="middle">{line}</text>'
        )

    plot_w = cell_w * len(stage_order)
    plot_h = cell_h * len(factor_order)
    body.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" rx="14" ry="14" fill="#ffffff" stroke="#d1d5db" stroke-width="1.5"/>'
    )

    max_abs_uplift = 0.0
    if mode == "uplift":
        values = [
            abs(value)
            for stage in stage_order
            for value in [matrix.get(stage, {}).get(factor) for factor in factor_order]
            if value is not None
        ]
        max_abs_uplift = max(values) if values else 0.0
        max_abs_uplift = max(max_abs_uplift, 0.05)

    for row_idx, factor in enumerate(factor_order):
        y = top + row_idx * cell_h
        factor_label = factor.replace("_", " ")
        body.append(
            f'<text class="tick" x="{left - 14}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="end">{factor_label}</text>'
        )

        for col_idx, stage in enumerate(stage_order):
            x = left + col_idx * cell_w
            value = matrix.get(stage, {}).get(factor)
            allowed = is_factor_allowed(stage, factor)
            if mode == "usage":
                fill = usage_color(value)
                display = "NA" if value is None else f"{value * 100:.0f}%"
            else:
                if not allowed:
                    fill = "#e5e7eb"
                    display = "N/A"
                else:
                    fill = uplift_color(value, max_abs_uplift)
                    display = "NA" if value is None else f"{value * 100:+.0f}%"

            body.append(
                f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="8" ry="8" fill="{fill}" stroke="#ffffff" stroke-width="1"/>'
            )
            body.append(
                f'<text class="cell-label" x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                f'text-anchor="middle" fill="{ideal_text_color(fill)}">{display}</text>'
            )

    for col_idx, stage in enumerate(stage_order):
        x = left + col_idx * cell_w + cell_w / 2
        counts = stage_counts.get(stage, {})
        lines = [
            STAGE_DISPLAY_NAMES.get(stage, stage.replace("_", " ").title()),
            f"n={counts.get('success_turns', 0)}/{counts.get('total_turns', 0)} success",
        ]
        base_y = top + plot_h + 24
        for line_idx, line in enumerate(lines):
            body.append(
                f'<text class="tick" x="{x:.1f}" y="{base_y + line_idx * 16:.1f}" text-anchor="middle">{line}</text>'
            )

    body.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 30:.1f}" text-anchor="middle">Stage</text>'
    )
    body.append(
        f'<text class="axis-label" x="36" y="{top + plot_h / 2:.1f}" transform="rotate(-90,36,{top + plot_h / 2:.1f})" text-anchor="middle">Factor</text>'
    )

    legend_x = left
    legend_y = height - 78
    if mode == "usage":
        draw_gradient_legend(
            body=body,
            x=legend_x,
            y=legend_y,
            width=340,
            height=18,
            title="Success-turn usage rate",
            labels=[(0.0, "0%"), (0.5, "50%"), (1.0, "100%")],
            gradient_id="usageLegend",
            stops=[
                ("#f8fbff", 0.0),
                ("#8ab4ff", 50.0),
                ("#1d4ed8", 100.0),
            ],
        )
        footnote = (
            "Definition: P(has_factor = 1 | successful turn, same model/stage/filter). "
            "Cells are NA when that stage has no successful turns."
        )
    else:
        max_pct = max_abs_uplift * 100
        draw_gradient_legend(
            body=body,
            x=legend_x,
            y=legend_y,
            width=420,
            height=18,
            title="Uplift over baseline usage",
            labels=[
                (0.0, f"-{max_pct:.0f}%"),
                (0.5, "0%"),
                (1.0, f"+{max_pct:.0f}%"),
            ],
            gradient_id="upliftLegend",
            stops=[
                ("#1d4ed8", 0.0),
                ("#f8fafc", 50.0),
                ("#c2410c", 100.0),
            ],
        )
        footnote = (
            "Definition: uplift = success-turn usage rate - same-stage allowed baseline. "
            "Gray cells are stage-disallowed factors. Positive cells mean the factor appears more often in successful turns than in allowed turns overall."
        )

    body.append(f'<text class="legend-text" x="{legend_x}" y="{height - 68:.1f}">{footnote}</text>')
    return svg_wrap(width, height, "\n".join(body))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Generated: {path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = read_round_factor_rows(csv_path)
    if not rows:
        raise ValueError("No round_factor rows found in the provided CSV.")

    target_model = resolve_target_model(rows, args.target_model)
    filtered = filter_rows(rows, target_model=target_model, cell_id=args.cell_id)
    if not filtered:
        raise ValueError("No rows left after applying target_model / cell_id filters.")

    stage_order = infer_stage_order(row.get("stage_name", "") for row in filtered)
    factor_order = infer_factor_order(filtered)
    stage_counts = compute_stage_counts(filtered, stage_order)
    success_usage, baseline_usage, global_allowed_baseline, uplift = compute_stage_factor_rates(
        filtered,
        stage_order=stage_order,
        factor_order=factor_order,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slugify(target_model)
    cell_slug = slugify(args.cell_id) if args.cell_id else "all_cells"
    stem = f"{model_slug}_{cell_slug}"

    success_csv = output_dir / f"{stem}_success_turn_usage.csv"
    baseline_csv = output_dir / f"{stem}_allowed_baseline_usage.csv"
    global_baseline_csv = output_dir / f"{stem}_global_allowed_baseline_usage.csv"
    uplift_csv = output_dir / f"{stem}_uplift.csv"
    counts_csv = output_dir / f"{stem}_stage_counts.csv"

    save_matrix_csv(success_csv, stage_order, factor_order, success_usage, "success-turn usage")
    save_matrix_csv(baseline_csv, stage_order, factor_order, baseline_usage, "same-stage allowed baseline usage")
    save_matrix_csv(global_baseline_csv, stage_order, factor_order, global_allowed_baseline, "global allowed baseline usage")
    save_matrix_csv(uplift_csv, stage_order, factor_order, uplift, "uplift")
    save_stage_counts_csv(counts_csv, stage_order, stage_counts)

    filter_note = f"cell_id = {args.cell_id}" if args.cell_id else "cell_id = all"
    success_svg = build_heatmap_svg(
        title="Stage x Factor Success-turn Usage Heatmap",
        subtitle_lines=[
            f"CSV: {csv_path.name}",
            f"target_model = {target_model} | {filter_note}",
        ],
        stage_order=stage_order,
        factor_order=factor_order,
        matrix=success_usage,
        stage_counts=stage_counts,
        mode="usage",
    )
    uplift_svg = build_heatmap_svg(
        title="Stage x Factor Uplift Heatmap",
        subtitle_lines=[
            f"CSV: {csv_path.name}",
            f"target_model = {target_model} | {filter_note}",
        ],
        stage_order=stage_order,
        factor_order=factor_order,
        matrix=uplift,
        stage_counts=stage_counts,
        mode="uplift",
    )

    save_text(output_dir / f"{stem}_success_turn_usage_heatmap.svg", success_svg)
    save_text(output_dir / f"{stem}_uplift_heatmap.svg", uplift_svg)


if __name__ == "__main__":
    main()
