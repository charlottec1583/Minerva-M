from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jailbreak_utils.ResultUtilsCustom import save_batch_summary


DEFAULT_STAGE_ORDER = [
    "context_establishment",
    "relationship_building",
    "constraint_induction",
    "escalation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a batch_summary_*.json from per-run attack_summary_*.json files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing attack_summary_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder to write reconstructed summary outputs",
    )
    return parser.parse_args()


def stage_to_index(stage_name: Optional[str], stage_order: List[str]) -> Optional[int]:
    if not stage_name:
        return None
    try:
        return stage_order.index(stage_name) + 1
    except ValueError:
        return None


def extract_first_success_stage(history: List[Dict[str, Any]]) -> Optional[str]:
    for record in history:
        if record.get("attack_is_success", False):
            return record.get("current_stage")
    return None


def parse_index_from_name(name: str) -> Optional[int]:
    patterns = [
        r"attack_summary_(?:ga_)?prompt_(\d+)_",
        r"ga_prompt_(\d+)_",
        r"prompt_(\d+)_",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def parse_cell_id_from_name(name: str) -> str:
    patterns = [
        r"attack_summary_(?:ga_)?prompt_\d+_([A-Z](?:_[A-Z]){3})_",
        r"(?:ga_)?prompt_\d+_([A-Z](?:_[A-Z]){3})_",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1)
    return ""


def parse_attack_summary(json_path: Path) -> Dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    history = data.get("history", []) or []
    models = metadata.get("models", {}) or {}

    index = parse_index_from_name(json_path.name)
    prompt = metadata.get("scenario_info") or metadata.get("intention") or ""
    success = str(metadata.get("final_status", "")).strip().lower() == "success"
    iterations = int(metadata.get("total_iterations", 0) or 0)

    final_score = 1
    if history:
        final_score = int(
            (history[-1].get("judge_result", {}) or {}).get("judgement_score", 1) or 1
        )

    first_success_stage = extract_first_success_stage(history)
    first_success_stage_index = stage_to_index(first_success_stage, DEFAULT_STAGE_ORDER)

    if history:
        planner_output = history[0].get("planner_output", {}) or {}
        cell_id = planner_output.get("cell_id") or parse_cell_id_from_name(json_path.name)
    else:
        cell_id = parse_cell_id_from_name(json_path.name)

    return {
        "index": index,
        "prompt": prompt,
        "success": success,
        "score": final_score,
        "iterations": iterations,
        "cell_id": cell_id,
        "first_success_stage": first_success_stage,
        "first_success_stage_index": first_success_stage_index,
        "target_model": models.get("target", ""),
        "planner_model": models.get("planner", ""),
        "generator_model": models.get("generator", ""),
        "judge_model": models.get("judge", ""),
        "factor_presence_rows": [],
        "html_file": json_path.with_suffix(".html").name,
        "json_file": json_path.name,
        "error": None,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    json_files = sorted(
        p for p in input_dir.glob("attack_summary_*.json") if p.is_file()
    )
    if not json_files:
        raise ValueError(f"No attack_summary_*.json files found in {input_dir}")

    results = [parse_attack_summary(path) for path in json_files]
    results = sorted(
        results,
        key=lambda x: (x.get("index") is None, x.get("index", 10**9), x.get("json_file", "")),
    )

    summary_path = save_batch_summary(results, str(output_dir))
    print(f"Rebuilt summary: {summary_path}")


if __name__ == "__main__":
    main()
