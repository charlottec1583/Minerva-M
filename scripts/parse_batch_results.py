#!/usr/bin/env python3
"""
Parse JSON files in batch_results folder, extract stage and ivs_activated information
"""
import json
import os
from pathlib import Path
from typing import Dict, Any


def parse_json_file(json_path: str) -> Dict[str, Any]:
    """Parse a single JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {
        'file_name': os.path.basename(json_path),
        'file_path': json_path,
        'total_stages': 0,
        'stages_used': [],
        'stage_executions': [],  # Record detailed info for each stage execution
        'ga_optimization_history': data.get('ga_optimization_history', []),  # Preserve GA optimization history
        'metadata': {
            'final_status': data.get('metadata', {}).get('final_status', 'unknown'),
            'total_iterations': data.get('metadata', {}).get('total_iterations', 0),
            'final_stage': data.get('metadata', {}).get('final_stage', 'unknown'),
            'target_model': data.get('metadata', {}).get('models', {}).get('target', 'unknown')
        }
    }

    # Get configured stages from the first history entry
    if data.get('history') and len(data['history']) > 0:
        first_entry = data['history'][0]
        experiment_config = first_entry.get('experiment_config', {})
        stages_config = experiment_config.get('stages', [])

        result['total_stages'] = len(stages_config)
        result['stages_used'] = [stage.get('stage') if isinstance(stage, dict) else stage for stage in stages_config]

    # Iterate through history to record each stage execution
    for entry in data.get('history', []):
        stage_execution = {
            'iteration': entry.get('current_iteration', -1),
            'stage': entry.get('current_stage', 'unknown'),
            'ivs_activated': entry.get('ivs_activated', []),
            'attack_is_success': entry.get('attack_is_success', False)
        }
        result['stage_executions'].append(stage_execution)

    return result


def parse_folder(folder_path: str) -> Dict[str, Any]:
    """Parse all JSON files in a folder"""
    folder_name = os.path.basename(folder_path.rstrip('/'))
    json_files = list(Path(folder_path).glob('*.json'))

    results = {
        'folder_name': folder_name,
        'folder_path': folder_path,
        'total_files': len(json_files),
        'files': []
    }

    for json_file in sorted(json_files):
        try:
            file_result = parse_json_file(str(json_file))

            # Skip invalid/meaningless files (empty stage executions, unknown metadata)
            if file_result.get('total_stages') == 0 and file_result.get('metadata', {}).get('total_iterations') == 0:
                print(f"Skipping invalid file: {json_file.name}")
                continue

            results['files'].append(file_result)
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            results['files'].append({
                'file_name': json_file.name,
                'error': str(e)
            })

    return results


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_batch_results.py <folder_path>")
        print("Example: python parse_batch_results.py /root/A_NEW_WORKSPACE/all4jailbreak/batch_results_20260330_112926/")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)

    result = parse_folder(folder_path)

    # Get script directory for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'parsed_results')
    os.makedirs(output_dir, exist_ok=True)

    # Output results to JSON file in script directory
    folder_basename = os.path.basename(folder_path.rstrip('/'))
    output_file = os.path.join(output_dir, f"parsed_results_{folder_basename}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")
    print(f"Total files found: {result['total_files']}")
    print(f"Valid files parsed: {len(result['files'])}")


if __name__ == '__main__':
    main()
