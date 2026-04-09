from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STAGE_DISPLAY_NAMES = {
    1: "Context Establishment",
    2: "Relationship Building",
    3: "Constraint Induction",
    4: "Escalation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-model cumulative success rate (C_k) comparison chart "
            "from one or more batch_summary_*.json files."
        )
    )
    parser.add_argument(
        "--summary",
        nargs="+",
        required=True,
        help="One or more batch_summary_*.json paths",
    )
    parser.add_argument(
        "--output-dir",
        default="presentation_charts/ck_comparison",
        help="Directory for generated SVG and CSV outputs",
    )
    parser.add_argument(
        "--title",
        default="Cumulative Success Rate by Stage (C_k)",
        help="Chart title",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Generated: {path}")


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
  .label {{ font-size: 12px; font-weight: 600; }}
  .legend-title {{ font-size: 14px; font-weight: 700; }}
  .legend-text {{ font-size: 12px; fill: #475569; }}
</style>
{body}
</svg>
"""


def model_color(index: int) -> str:
    palette = [
        "#ea5a5a",
        "#7fbf7b",
        "#f5c04a",
        "#c18a67",
        "#b3b3b3",
    ]
    return palette[index % len(palette)]


def model_color_by_name(model_name: str, fallback_index: int) -> str:
    lower = model_name.lower()
    if "qwen" in lower:
        return "#ea5a5a"
    if "deepseek" in lower:
        return "#7fbf7b"
    if "gemini" in lower:
        return "#f5c04a"
    if "gpt-3.5" in lower or "gpt 3.5" in lower or "turbo" in lower:
        return "#c18a67"
    if "gpt-5.1" in lower or "gpt 5.1" in lower:
        return "#b3b3b3"
    return model_color(fallback_index)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_single_model_name(summary: Dict, fallback: str) -> str:
    stage_by_model = summary.get("stage_metrics_by_model") or {}
    if len(stage_by_model) == 1:
        return next(iter(stage_by_model.keys()))

    results = summary.get("results") or []
    names = []
    seen = set()
    for item in results:
        name = str(item.get("target_model", "")).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if len(names) == 1:
        return names[0]
    return fallback


def collect_model_metrics(summary_path: Path) -> List[Dict]:
    summary = load_json(summary_path)
    models = []
    stage_metrics_by_model = summary.get("stage_metrics_by_model") or {}

    if stage_metrics_by_model:
        for model_name, metrics in stage_metrics_by_model.items():
            models.append(
                {
                    "model": model_name,
                    "metrics": metrics,
                    "summary_path": str(summary_path),
                }
            )
        return models

    fallback_name = infer_single_model_name(summary, summary_path.stem)
    metrics = summary.get("stage_metrics") or {}
    return [
        {
            "model": fallback_name,
            "metrics": metrics,
            "summary_path": str(summary_path),
        }
    ]


def unique_model_records(records: Sequence[Dict]) -> List[Dict]:
    by_model: Dict[str, Dict] = {}
    duplicates = set()
    for record in records:
        model = record["model"]
        if model in by_model:
            duplicates.add(model)
            continue
        by_model[model] = record

    if duplicates:
        dup_text = ", ".join(sorted(duplicates))
        raise ValueError(
            f"Duplicate model names found across summaries: {dup_text}. "
            "Please keep one summary per model or rename models upstream."
        )
    return list(by_model.values())


def write_metrics_csv(output_path: Path, records: Sequence[Dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target_model",
                "source_summary",
                "N",
                "ASR",
                "AUC_C",
                "C1",
                "C2",
                "C3",
                "C4",
                "Delta1",
                "Delta2",
                "Delta3",
                "Delta4",
                "ratio_c3_c4",
                "delta4",
                "earliest_converge_stage",
                "decision",
            ]
        )
        for record in records:
            metrics = record["metrics"]
            ck = metrics.get("C_k", [])
            dk = metrics.get("Delta_k", [])
            conv = metrics.get("convergence", {})
            writer.writerow(
                [
                    record["model"],
                    record["summary_path"],
                    metrics.get("total"),
                    metrics.get("ASR"),
                    metrics.get("AUC_C"),
                    ck[0] if len(ck) > 0 else None,
                    ck[1] if len(ck) > 1 else None,
                    ck[2] if len(ck) > 2 else None,
                    ck[3] if len(ck) > 3 else None,
                    dk[0] if len(dk) > 0 else None,
                    dk[1] if len(dk) > 1 else None,
                    dk[2] if len(dk) > 2 else None,
                    dk[3] if len(dk) > 3 else None,
                    conv.get("ratio_c3_c4"),
                    conv.get("delta4"),
                    conv.get("earliest_converge_stage"),
                    conv.get("decision"),
                ]
            )
    print(f"Generated: {output_path}")


def build_ck_svg(title: str, records: Sequence[Dict]) -> str:
    width, height = 1500, 860
    left, right, top, bottom = 100, 420, 145, 170
    plot_w = width - left - right
    plot_h = height - top - bottom
    stages = [1, 2, 3, 4]
    y_min = 0.25
    y_max = 1.00

    def x_of(stage: int) -> float:
        return left + (stage - 1) * (plot_w / 3)

    def y_of(value: float) -> float:
        clamped = min(max(value, y_min), y_max)
        return top + (y_max - clamped) / (y_max - y_min) * plot_h

    body: List[str] = []
    body.append(f'<text class="title" x="{width / 2:.1f}" y="58" text-anchor="middle">{title}</text>')
    body.append(
        f'<text class="subtitle" x="{width / 2:.1f}" y="84" text-anchor="middle">Compare C1-C4 across target models to see whether the curves flatten after a specific stage.</text>'
    )

    for tick in [0.25, 0.50, 0.75, 1.00]:
        y = y_of(tick)
        pct = int(round(tick * 100))
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        body.append(f'<text class="tick" x="{left - 12}" y="{y + 5:.1f}" text-anchor="end">{pct}%</text>')

    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#0f172a" stroke-width="2"/>')
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#0f172a" stroke-width="2"/>')

    for stage in stages:
        x = x_of(stage)
        body.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#0f172a" stroke-width="1.5"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 26:.1f}" text-anchor="middle">Stage {stage}:</text>')
        body.append(
            f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 44:.1f}" text-anchor="middle">{STAGE_DISPLAY_NAMES.get(stage, "")}</text>'
        )

    body.append(f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 78:.1f}" text-anchor="middle">Stage k</text>')
    body.append(
        f'<text class="axis-label" x="32" y="{top + plot_h / 2:.1f}" transform="rotate(-90,32,{top + plot_h / 2:.1f})" text-anchor="middle">Cumulative Success Rate (C_k)</text>'
    )

    legend_x = left + plot_w + 28
    legend_y = top + 24
    body.append(f'<text class="legend-title" x="{legend_x}" y="{legend_y - 8:.1f}">Models</text>')

    stage_label_offsets = {
        1: [(48, -30), (48, -10), (48, 10), (48, 30), (48, 50)],
        2: [(-48, -30), (48, -10), (-48, 10), (48, 30), (0, 50)],
        3: [(-56, -36), (56, -18), (-56, 0), (56, 18), (0, 40)],
        4: [(-48, -30), (-48, -10), (-48, 10), (-48, 30), (-48, 50)],
    }

    stage_label_positions: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for stage in stages:
        sortable: List[Tuple[int, float]] = []
        for idx, record in enumerate(records):
            ck = record["metrics"].get("C_k", [])
            if stage <= len(ck):
                sortable.append((idx, float(ck[stage - 1])))
        sortable.sort(key=lambda item: item[1], reverse=True)
        offsets = stage_label_offsets.get(stage, [(48, 0)] * max(1, len(sortable)))
        for rank, (record_idx, val) in enumerate(sortable):
            dx, dy = offsets[min(rank, len(offsets) - 1)]
            x = x_of(stage)
            y = y_of(val)
            stage_label_positions[(record_idx, stage)] = (x + dx, y + dy)

    for idx, record in enumerate(records):
        color = model_color_by_name(record["model"], idx)
        model = record["model"]
        metrics = record["metrics"]
        ck = metrics.get("C_k", [])
        conv = metrics.get("convergence", {})

        points = []
        for stage, val in enumerate(ck, start=1):
            points.append(f"{x_of(stage):.1f},{y_of(float(val)):.1f}")

        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        )

        for stage, val in enumerate(ck, start=1):
            x = x_of(stage)
            y = y_of(float(val))
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            label_x, label_y = stage_label_positions.get((idx, stage), (x, y - 12))
            label_text = f"{float(val) * 100:.0f}%"
            box_w = 40
            box_h = 20
            body.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{label_x:.1f}" y2="{label_y - 6:.1f}" stroke="{color}" stroke-width="1.1" opacity="0.75"/>'
            )
            body.append(
                f'<rect x="{label_x - box_w / 2:.1f}" y="{label_y - box_h + 2:.1f}" width="{box_w}" height="{box_h}" rx="8" ry="8" fill="#ffffff" stroke="{color}" stroke-width="1.2" opacity="0.95"/>'
            )
            body.append(
                f'<text class="label" x="{label_x:.1f}" y="{label_y - 4:.1f}" text-anchor="middle" fill="{color}">{label_text}</text>'
            )

        end_y = legend_y + idx * 74 + 24
        decision = conv.get("decision", "not_converged")
        earliest = conv.get("earliest_converge_stage")
        if earliest is not None:
            decision_text = f"first converge: Stage {earliest}"
        else:
            decision_text = "first converge: none"
        ratio = conv.get("ratio_c3_c4")
        delta4 = conv.get("delta4")
        ratio_text = f"C3/C4={ratio:.2f}" if isinstance(ratio, (int, float)) else "C3/C4=N/A"
        delta_text = f"Delta4={delta4:.2f}" if isinstance(delta4, (int, float)) else "Delta4=N/A"

        body.append(f'<line x1="{legend_x}" y1="{end_y - 10:.1f}" x2="{legend_x + 36}" y2="{end_y - 10:.1f}" stroke="{color}" stroke-width="4"/>')
        body.append(f'<circle cx="{legend_x + 18:.1f}" cy="{end_y - 10:.1f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
        body.append(f'<text class="legend-text" x="{legend_x + 48}" y="{end_y - 4:.1f}">{model}</text>')
        body.append(
            f'<text class="legend-text" x="{legend_x + 48}" y="{end_y + 14:.1f}">{ratio_text}, {delta_text}</text>'
        )
        body.append(
            f'<text class="legend-text" x="{legend_x + 48}" y="{end_y + 30:.1f}">{decision_text}</text>'
        )

    body.append(
        f'<text class="subtitle" x="{left + plot_w / 2:.1f}" y="{height - 42:.1f}" text-anchor="middle">Reading guide: earlier flattening suggests earlier saturation; the legend reports the first stage at which the remaining tail gain becomes negligible under the convergence rule.</text>'
    )
    return svg_wrap(width, height, "\n".join(body))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for summary_arg in args.summary:
        summary_path = Path(summary_arg).expanduser()
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        records.extend(collect_model_metrics(summary_path))

    records = unique_model_records(records)
    records = sorted(records, key=lambda item: item["model"].lower())

    csv_path = output_dir / "ck_comparison_metrics.csv"
    write_metrics_csv(csv_path, records)

    svg_path = output_dir / "ck_comparison.svg"
    save_text(svg_path, build_ck_svg(args.title, records))


if __name__ == "__main__":
    main()
