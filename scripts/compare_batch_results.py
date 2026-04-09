#!/usr/bin/env python3
"""
Compare two parsed batch results and generate visualization charts
"""
import json
import os
import sys
import argparse
from typing import Dict, List, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def load_parsed_data(json_path: str) -> Dict[str, Any]:
    """Load parsed results JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_successful_file(file_data: Dict[str, Any]) -> bool:
    """Return True when a parsed result reached a successful attack round."""
    if 'error' in file_data:
        return False

    final_status = str(file_data.get('metadata', {}).get('final_status', '')).strip().lower()
    if final_status:
        return final_status == 'success'

    return any(stage.get('attack_is_success', False) for stage in file_data.get('stage_executions', []))


def get_success_stage_count(file_data: Dict[str, Any]) -> int:
    """Count stage executions only up to and including the first success round."""
    stage_executions = file_data.get('stage_executions', [])
    for idx, stage_exec in enumerate(stage_executions, start=1):
        if stage_exec.get('attack_is_success', False):
            return idx
    return len(stage_executions)


def calculate_stage_executions_stats(parsed_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate stage execution stats using successful queries only."""
    execution_counts = []

    for file_data in parsed_data.get('files', []):
        if is_successful_file(file_data):
            count = get_success_stage_count(file_data)
            if count > 0:
                execution_counts.append(count)

    if not execution_counts:
        return {'mean': 0, 'std': 0, 'count': 0}

    return {
        'mean': np.mean(execution_counts),
        'std': np.std(execution_counts),
        'count': len(execution_counts)
    }


def calculate_stage_ivs_frequency(parsed_data: Dict[str, Any]) -> defaultdict:
    """Calculate IVs frequency for each stage"""
    stage_ivs = defaultdict(Counter)

    for file_data in parsed_data.get('files', []):
        if 'error' in file_data:
            continue

        for stage_exec in file_data.get('stage_executions', []):
            stage = stage_exec.get('stage', 'unknown')
            ivs = stage_exec.get('ivs_activated', [])
            stage_ivs[stage].update(ivs)

    return stage_ivs


def get_model_icon_path(model_name: str) -> str:
    """Get icon path for a model, with fallbacks"""
    # Get script directory and icon folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_dir = os.path.join(script_dir, 'icons')

    model_lower = model_name.lower()

    # Map variations to base names
    name_mapping = {
        'o3': os.path.join(icon_dir, 'openai.png'),
        'gpt': os.path.join(icon_dir, 'openai.png'),
        'chatgpt': os.path.join(icon_dir, 'openai.png'),
        'qwen': os.path.join(icon_dir, 'qwen-color.png'),
        'deepseek': os.path.join(icon_dir, 'deepseek-color.png'),
        'gemini': os.path.join(icon_dir, 'gemini-color.png'),
        'claude': os.path.join(icon_dir, 'claude-color.png'),
        'minimax': os.path.join(icon_dir, 'minimax-color.png'),
        'moonshot': os.path.join(icon_dir, 'moonshot.png'),
        'kimi': os.path.join(icon_dir, 'moonshot.png'),
    }

    # Try to find matching key in name_mapping
    for key, icon_path in name_mapping.items():
        if key in model_lower or model_lower in key:
            if os.path.exists(icon_path):
                return icon_path

    return None


def shorten_label(label: str, max_words_per_line: int = 3) -> str:
    """
    Shorten label to at most max_words_per_line words per line.
    Preserves newlines for multi-line labels.
    If the truncated label ends with a number, keeps one more word if it's also a number.
    """
    lines = label.split('\n')
    shortened_lines = []

    for line in lines:
        words = line.replace('-', ' ').title().split()
        if len(words) > max_words_per_line:
            words = words[:max_words_per_line]
            # If the last word is a number, try to keep one more word if it's also a number
            # Skip years starting with "20" (e.g., 2024, 2025)
            def is_year_starting_with_20(word: str) -> bool:
                """Check if a word is a year starting with '20'"""
                word_without_dot = word.replace('.', '', 1)
                return word_without_dot.isdigit() and word_without_dot.startswith('20')

            if words and words[-1].replace('.', '', 1).isdigit() and not is_year_starting_with_20(words[-1]):
                next_idx = max_words_per_line
                original_words = line.replace('-', ' ').title().split()
                if (next_idx < len(original_words) and
                    original_words[next_idx].replace('.', '', 1).isdigit() and
                    not is_year_starting_with_20(original_words[next_idx])):
                    words.append(original_words[next_idx])
        shortened_lines.append(' '.join(words))

    return '\n'.join(shortened_lines)


def clean_model_name(model_name: str) -> str:
    """
    Remove domain prefix from model name (e.g., aws/gpt-4 -> gpt-4)
    """
    if '/' in model_name:
        return model_name.split('/')[-1]
    return model_name


def calculate_success_rate(parsed_data: Dict[str, Any]) -> float:
    """Calculate success rate from parsed data"""
    success_count = 0
    total_count = 0

    for file_data in parsed_data.get('files', []):
        if 'error' not in file_data:
            total_count += 1
            if file_data.get('metadata', {}).get('final_status') == 'success':
                success_count += 1

    if total_count == 0:
        return 0.0
    return success_count / total_count


def calculate_population_stats(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate population statistics using successful queries only."""
    population_counts = []

    for file_data in parsed_data.get('files', []):
        if is_successful_file(file_data):
            ga_history = file_data.get('ga_optimization_history', [])
            total_pop = 0
            for generation in ga_history:
                population = generation.get('population', [])
                total_pop += len(population)
            population_counts.append(total_pop)

    if not population_counts:
        return {'mean': 0, 'std': 0, 'total': 0, 'count': 0}

    return {
        'mean': np.mean(population_counts),
        'std': np.std(population_counts),
        'total': sum(population_counts),
        'count': len(population_counts)
    }


def plot_success_rates(data_list: List[Dict], labels: List[str], output_dir: str):
    """Plot success rate comparison bar chart with model icons"""
    n_models = len(data_list)
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 7))

    # Calculate success rates
    rates = [calculate_success_rate(data) * 100 for data in data_list]

    # Use colormap for N models
    cmap = plt.colormaps.get_cmap('Set1').resampled(n_models)
    colors = [cmap(i) for i in range(n_models)]

    # Create bars
    x = np.arange(n_models)
    width = 0.8
    bars = ax.bar(x, rates, width, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add model icons above bars
    for i, (model, rate) in enumerate(zip(labels, rates)):
        # Get icon path
        icon_path = get_model_icon_path(model)

        if icon_path and os.path.exists(icon_path):
            # Load and add icon
            try:
                img = Image.open(icon_path)
                # Convert to RGBA to fix palette mode RGB issues
                if img.mode == 'P':
                    img = img.convert('RGBA')
                # Resize icon with larger size for better clarity
                img = img.resize((60, 60), Image.Resampling.LANCZOS)
                # Increase zoom for better display quality
                imagebox = OffsetImage(img, zoom=0.8, interpolation='bilinear')
                ab = AnnotationBbox(imagebox, (i, rate + 8.1),
                                 xybox=(0, 0),
                                 xycoords='data',
                                 boxcoords="offset points",
                                 box_alignment=(0.5, 0),
                                 pad=0,
                                 frameon=False)  # Remove frame/border
                ax.add_artist(ab)
            except Exception as e:
                # Fallback to text if icon fails
                ax.text(i, rate + 3, model.replace('-', ' ').title(),
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       rotation=0)
        else:
            # Fallback to model name text
            ax.text(i, rate + 3, model.replace('-', ' ').title(),
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   rotation=0)

        # Add percentage value on bar
        height = bars[i].get_height()
        ax.text(i, height + 1, f'{rate:.1f}%',
               ha='center', va='bottom', fontsize=20 if n_models > 3 else 16, fontweight='bold', color='black')

    ax.set_ylabel('Success Rate (%)', fontsize=20, fontweight='bold')
    ax.set_title('Attack Success Rate Comparison', fontsize=20, fontweight='bold', pad=60)
    # Set y-axis upper limit to max value rounded up to nearest 10
    y_max = int(np.ceil(max(rates) / 10) * 10) + 9
    ax.set_ylim(0, y_max)

    # Make the 50% tick label bold
    ytick_labels = ax.get_yticklabels()
    for label in ytick_labels:
        if label.get_text() == '50':
            label.set_fontweight('bold')

    ax.set_xticks(x)
    ax.set_xticklabels([shorten_label(model) for model in labels],
                       fontsize=20 if n_models > 3 else 16, fontweight='bold', rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'success_rate_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_stage_executions(data_list: List[Dict], labels: List[str], output_dir: str):
    """Plot stage execution statistics for successful queries with error bars and model icons."""
    n_models = len(data_list)
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 7))

    stats_list = [calculate_stage_executions_stats(data) for data in data_list]

    means = [s['mean'] for s in stats_list]
    stds = [s['std'] for s in stats_list]

    # Use colormap for N models
    cmap = plt.colormaps.get_cmap('Set1').resampled(n_models)
    colors = [cmap(i) for i in range(n_models)]

    x_pos = np.arange(n_models)

    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5, capsize=10, error_kw={'linewidth': 2})

    # Add model icons above bars
    for i, (model, mean, std) in enumerate(zip(labels, means, stds)):
        # Get icon path
        icon_path = get_model_icon_path(model)

        if icon_path and os.path.exists(icon_path):
            # Load and add icon
            try:
                img = Image.open(icon_path)
                # Convert to RGBA to fix palette mode RGB issues
                if img.mode == 'P':
                    img = img.convert('RGBA')
                # Resize icon with larger size for better clarity
                img = img.resize((60, 60), Image.Resampling.LANCZOS)
                # Increase zoom for better display quality
                imagebox = OffsetImage(img, zoom=0.8, interpolation='bilinear')
                ab = AnnotationBbox(imagebox, (i, mean + std + 0.3),
                                 xybox=(0, 0),
                                 xycoords='data',
                                 boxcoords="offset points",
                                 box_alignment=(0.5, 0),
                                 pad=0,
                                 frameon=False)  # Remove frame/border
                ax.add_artist(ab)
            except Exception as e:
                # Fallback to text if icon fails
                ax.text(i, mean + std + 0.3, model.replace('-', ' ').title(),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # Fallback to model name text
            ax.text(i, mean + std + 0.3, model.replace('-', ' ').title(),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add value label on bar (above error bar)
        ax.text(i, mean + std + 0.05, f'{mean:.2f}±{std:.2f}',
               ha='center', va='bottom', fontsize=14 if n_models > 3 else 16, fontweight='bold', color='black')

    ax.set_ylabel('Average Stage Executions', fontsize=20, fontweight='bold')
    ax.set_title('Stage Execution Statistics (Mean ± Std)', fontsize=20, fontweight='bold', pad=60)
    ax.set_ylabel('Average Stage Executions', fontsize=20, fontweight='bold')
    ax.set_title('Stage Execution Statistics (Successful Queries Only, Mean ± Std)', fontsize=20, fontweight='bold', pad=60)
    ax.set_title('Stage Execution Statistics (Successful Queries Only, Mean +/- Std)', fontsize=20, fontweight='bold', pad=60)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([shorten_label(model) for model in labels],
                       fontsize=18 if n_models > 3 else 20, fontweight='bold', rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set fixed y-axis range for stage executions
    ax.set_ylim(0, 4.9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'stage_executions_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_stage_ivs_usage(data_list: List[Dict], labels: List[str], output_dir: str):
    """Plot IVs usage by stage for all models"""
    n_models = len(data_list)
    stage_ivs_list = [calculate_stage_ivs_frequency(data) for data in data_list]

    # Get total file counts for percentage calculation
    total_files_list = [len([f for f in data.get('files', []) if 'error' not in f]) for data in data_list]

    # Get correct stage order from parsed data
    stage_order = None
    for data in data_list:
        for file_data in data.get('files', []):
            if 'error' not in file_data and 'stages_used' in file_data:
                stage_order = file_data['stages_used']
                break
        if stage_order:
            break

    # Get all unique stages
    all_stage_keys = set()
    for stage_ivs in stage_ivs_list:
        all_stage_keys.update(stage_ivs.keys())
    all_stages = sorted(list(all_stage_keys))

    # Sort stages by correct order if available, otherwise keep alphabetical
    if stage_order:
        # Create a mapping of stage name to its index in stage_order
        stage_order_map = {stage: idx for idx, stage in enumerate(stage_order)}
        # Filter to only stages that exist in our data
        valid_stages = [s for s in stage_order if s in all_stages]
        # Add any remaining stages not in stage_order (alphabetically at the end)
        remaining_stages = sorted([s for s in all_stages if s not in stage_order_map])
        all_stages = valid_stages + remaining_stages

    # Get all unique IVs
    all_ivs = set()
    for stage_ivs in stage_ivs_list:
        for ivs in stage_ivs.values():
            all_ivs.update(ivs.keys())
    all_ivs = sorted(list(all_ivs))

    # Use colormap for N models
    cmap = plt.colormaps.get_cmap('Set1').resampled(n_models)
    colors = [cmap(i) for i in range(n_models)]

    # Create subplots for each stage (horizontal layout)
    n_stages = len(all_stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 6))
    if n_stages == 1:
        axes = np.array([axes])

    for idx, stage in enumerate(all_stages):
        ax = axes[idx]

        # Get IVs for this stage from all models
        stage_ivs_counters = [stage_ivs.get(stage, Counter()) for stage_ivs in stage_ivs_list]

        # Get all IVs used by any model for this stage
        all_ivs_stage = set()
        for counter in stage_ivs_counters:
            all_ivs_stage.update(counter.keys())

        if not all_ivs_stage:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            stage_name = stage.replace('_', ' ').title()
            ax.set_title(f'Stage {idx + 1}: {stage_name}', fontweight='bold')
            continue

        # Sort by total frequency across all models
        ivs_sorted = sorted(all_ivs_stage,
                           key=lambda x: sum(counter.get(x, 0) for counter in stage_ivs_counters),
                           reverse=True)

        x = np.arange(len(ivs_sorted))
        width = 0.8 / n_models  # Adjust width based on number of models

        # Convert counts to percentages and create bars
        for i, (ivs_counter, total_files, label, color) in enumerate(zip(stage_ivs_counters, total_files_list, labels, colors)):
            counts = [ivs_counter.get(iv, 0) / total_files * 100 if total_files > 0 else 0 for iv in ivs_sorted]
            offset = width * (i - (n_models - 1) / 2)
            bars = ax.bar(x + offset, counts, width, label=shorten_label(label),
                          color=color, alpha=0.7, edgecolor='black', linewidth=1)

            # # Add text labels on top of bars
            # for bar, count in zip(bars, counts):
            #     if count > 0:
            #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            #                f'{count:.1f}', ha='center', va='bottom', fontsize=8, color='black')

        # ax.set_xlabel('IVs', fontsize=12, fontweight='bold')  # Remove x-label
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        stage_name = stage.replace('_', ' ').title()
        # Only show ylabel on first subplot
        if idx > 0:
            ax.set_ylabel('')
        ax.set_title(f'Stage {idx + 1}: {stage_name} (%)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([iv.replace('_', ' ') for iv in ivs_sorted],
                          rotation=45, ha='right', fontsize=10)
        # Only show legend on the last subplot
        if idx == len(all_stages) - 1:
            ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Hide extra subplots
    for idx in range(len(all_stages), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07)  # Reduce spacing between subplots even more
    output_path = os.path.join(output_dir, 'stage_ivs_usage_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_population_per_query(data_list: List[Dict], labels: List[str], output_dir: str):
    """Plot average population per successful query with error bars (std dev)."""
    n_models = len(data_list)
    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 7))

    stats_list = [calculate_population_stats(data) for data in data_list]

    means = [s['mean'] for s in stats_list]
    stds = [s['std'] for s in stats_list]

    # Use colormap for N models
    cmap = plt.colormaps.get_cmap('Set1').resampled(n_models)
    colors = [cmap(i) for i in range(n_models)]

    x_pos = np.arange(n_models)

    bars = ax.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5, capsize=10, error_kw={'linewidth': 2})

    # Add model icons above bars
    for i, (model, mean, std) in enumerate(zip(labels, means, stds)):
        # Get icon path
        icon_path = get_model_icon_path(model)

        if icon_path and os.path.exists(icon_path):
            # Load and add icon
            try:
                img = Image.open(icon_path)
                # Convert to RGBA to fix palette mode RGB issues
                if img.mode == 'P':
                    img = img.convert('RGBA')
                # Resize icon with larger size for better clarity
                img = img.resize((60, 60), Image.Resampling.LANCZOS)
                # Increase zoom for better display quality
                imagebox = OffsetImage(img, zoom=0.8, interpolation='bilinear')
                ab = AnnotationBbox(imagebox, (i, mean + std + (max(means) * 0.15)),
                                 xybox=(0, 0),
                                 xycoords='data',
                                 boxcoords="offset points",
                                 box_alignment=(0.5, 0),
                                 pad=0,
                                 frameon=False)  # Remove frame/border
                ax.add_artist(ab)
            except Exception as e:
                # Fallback to text if icon fails
                ax.text(i, mean + std + (max(means) * 0.03), model.replace('-', ' ').title(),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # Fallback to model name text
            ax.text(i, mean + std + (max(means) * 0.03), model.replace('-', ' ').title(),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add value label on bar (above error bar)
        ax.text(i, mean + std + (max(means) * 0.01), f'{mean:.2f}±{std:.2f}',
               ha='center', va='bottom', fontsize=14 if n_models > 3 else 16, fontweight='bold', color='black')

    ax.set_ylabel('Average Population Per Query', fontsize=20, fontweight='bold')
    ax.set_title('GA Population Per Query (Mean ± Std)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Population Per Successful Query', fontsize=20, fontweight='bold')
    ax.set_title('GA Population Per Successful Query (Mean ± Std)', fontsize=20, fontweight='bold')
    ax.set_title('GA Population Per Successful Query (Mean +/- Std)', fontsize=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([shorten_label(model) for model in labels],
                       fontsize=20 if n_models > 3 else 16, fontweight='bold', rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set ylim to ensure icons are fully visible
    max_value = max([m + s for m, s in zip(means, stds)]) if means else 1
    ax.set_ylim(0, max_value * 1.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'population_per_query_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_stage_ivs_heatmap(data_list: List[Dict], labels: List[str], output_dir: str):
    """Plot IVs usage heatmap by stage for all models"""
    n_models = len(data_list)
    stage_ivs_list = [calculate_stage_ivs_frequency(data) for data in data_list]

    # Get correct stage order from parsed data
    stage_order = None
    for data in data_list:
        for file_data in data.get('files', []):
            if 'error' not in file_data and 'stages_used' in file_data:
                stage_order = file_data['stages_used']
                break
        if stage_order:
            break

    # Get all unique stages
    all_stage_keys = set()
    for stage_ivs in stage_ivs_list:
        all_stage_keys.update(stage_ivs.keys())
    all_stages = sorted(list(all_stage_keys))

    # Sort stages by correct order if available, otherwise keep alphabetical
    if stage_order:
        # Create a mapping of stage name to its index in stage_order
        stage_order_map = {stage: idx for idx, stage in enumerate(stage_order)}
        # Filter to only stages that exist in our data
        valid_stages = [s for s in stage_order if s in all_stages]
        # Add any remaining stages not in stage_order (alphabetically at the end)
        remaining_stages = sorted([s for s in all_stages if s not in stage_order_map])
        all_stages = valid_stages + remaining_stages

    # Get all unique IVs
    all_ivs = set()
    for stage_ivs in stage_ivs_list:
        for ivs in stage_ivs.values():
            all_ivs.update(ivs.keys())
    all_ivs = sorted(list(all_ivs))

    # Get total file counts for percentage calculation
    total_files_list = [len([f for f in data.get('files', []) if 'error' not in f]) for data in data_list]

    # Create matrices for heatmaps (with percentages)
    matrices = []
    for idx, stage_ivs in enumerate(stage_ivs_list):
        matrix = []
        total_files = total_files_list[idx]
        for stage in all_stages:
            row = []
            for iv in all_ivs:
                count = stage_ivs.get(stage, Counter()).get(iv, 0)
                # Convert to percentage
                percentage = (count / total_files * 100) if total_files > 0 else 0
                row.append(percentage)
            matrix.append(row)
        matrices.append(np.array(matrix))

    # Create horizontal layout for subplots
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))

    # Handle single model case
    if n_models == 1:
        axes = np.array([axes])

    # Define colormaps for different models
    cmaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'YlOrBr', 'YlGn', 'GnBu', 'YlGnBu', 'PuRd']

    for idx, (label, matrix, ax) in enumerate(zip(labels, matrices, axes)):
        if idx >= n_models:
            ax.set_visible(False)
            continue

        cmap = cmaps[idx % len(cmaps)]

        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set_xticks(np.arange(len(all_ivs)))
        ax.set_yticks(np.arange(len(all_stages)))
        ax.set_xticklabels([iv.replace('_', ' ') for iv in all_ivs], rotation=45, ha='right', fontsize=8)
        # Only show yticklabels for the first subplot
        if idx == 0:
            ax.set_yticklabels([f'Stage {i+1}' for i in range(len(all_stages))], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(f'{label}: IVs Usage by Stage (%)', fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(len(all_stages)):
            for j in range(len(all_ivs)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=7)

        # Only add label to the last colorbar
        if idx == n_models - 1:
            plt.colorbar(im, ax=ax, label='Percentage (%)', fraction=0.046, pad=0.04)
        else:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.07)  # Reduce spacing between subplots even more
    output_path = os.path.join(output_dir, 'stage_ivs_heatmap_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def parse_json_with_suffixes(args_list: list) -> List[tuple]:
    """
    Parse command line arguments to extract JSON file paths and their optional suffixes.

    Args:
        args_list: List of command line arguments

    Returns:
        List of tuples (json_path, suffix) where suffix can be None

    Example:
        Input: ['file1.json', 'w/ xx', 'file2.json', 'file3.json', 'w/ yy']
        Output: [('file1.json', 'w/ xx'), ('file2.json', None), ('file3.json', 'w/ yy')]
    """
    result = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]

        # Check if it's a JSON file
        if arg.endswith('.json'):
            json_path = arg
            # Look ahead to see if the next argument is a suffix (not a JSON file)
            suffix = None
            if i + 1 < len(args_list):
                next_arg = args_list[i + 1]
                if not next_arg.endswith('.json'):
                    suffix = next_arg
                    i += 1  # Skip the next argument
            result.append((json_path, suffix))
        else:
            print(f"Warning: Unexpected argument '{arg}' (expected a .json file)")
        i += 1

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Compare parsed batch results and generate visualization charts'
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='Parsed result JSON files to compare. Each JSON file can be followed by an optional suffix label.'
    )
    parser.add_argument(
        '-o', '--output-name',
        default='comparison_charts',
        help='Output directory name (default: comparison_charts)'
    )

    args = parser.parse_args()

    # Parse JSON files with optional suffixes
    json_with_suffixes = parse_json_with_suffixes(args.json_files)

    # Validate all files exist
    for json_path, _ in json_with_suffixes:
        if not os.path.isfile(json_path):
            print(f"Error: {json_path} does not exist")
            sys.exit(1)

    # Load all data
    print(f"Loading {len(json_with_suffixes)} parsed results...")
    data_list = []
    labels = []
    for json_path, suffix in json_with_suffixes:
        data = load_parsed_data(json_path)
        data_list.append(data)

        # Extract model name from metadata
        model_name = data.get('files', [{}])[0].get('metadata', {}).get('target_model', f'Model {len(data_list)}')
        # Remove domain prefix (e.g., aws/gpt-4 -> gpt-4)
        model_name = clean_model_name(model_name)

        # Combine model name with suffix if provided
        # Use newline for better display in charts
        if suffix:
            label = f"{model_name}\n{suffix}"
        else:
            label = model_name
        labels.append(label)

    print(f"\nComparing {len(data_list)} models:")
    for i, (label, data) in enumerate(zip(labels, data_list)):
        print(f"  {i+1}. {label}: {data.get('total_files', 0)} files")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating charts...")
    print("="*60)

    # Generate plots
    plot_success_rates(data_list, labels, output_dir)
    plot_stage_executions(data_list, labels, output_dir)
    plot_population_per_query(data_list, labels, output_dir)
    plot_stage_ivs_usage(data_list, labels, output_dir)
    plot_stage_ivs_heatmap(data_list, labels, output_dir)

    print("="*60)
    print(f"\nAll charts saved to: {output_dir}")


if __name__ == '__main__':
    main()
