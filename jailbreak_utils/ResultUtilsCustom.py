from typing import Any, Dict, List, Optional
import json
import os.path as osp
from datetime import datetime
from pathlib import Path
from string import Template
from jailbreak_core.AttackWorkflow import AttackState


# ============================================================================
# Utilities for HTML Preview
# ============================================================================

_TEMPLATE_DIR = Path(__file__).parent / "html_templates"


def _get_template_dir() -> Path:
    """Get the directory containing HTML templates."""
    return _TEMPLATE_DIR


def _load_template(template_name: str) -> str:
    """Load an HTML template file.

    Args:
        template_name: Name of the template file (e.g., 'summary_template.html')

    Returns:
        The template content as a string.
    """
    template_path = _get_template_dir() / template_name
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not isinstance(text, str):
        text = str(text)
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;"))


def compute_stage_metrics(results: List[Dict[str, Any]], stage_count: int = 4) -> Dict[str, Any]:
    """Compute stage-based metrics from batch results.

    Expects each result to include:
      - success (bool)
      - iterations (int)
    """
    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    failed = total - successful
    avg_iterations = sum(r.get("iterations", 0) for r in results) / total if total > 0 else 0

    # Simplified metrics without first-success-stage tracking
    asr = successful / total if total > 0 else 0.0

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "avg_iterations": avg_iterations,
        "ASR": asr,
    }


def save_summary_html(
    state: AttackState,
    filepath: Optional[str] = None,
    model_info: Optional[Dict[str, str]] = None,
) -> str:
    """Save attack workflow summary as HTML with interactive visualization.

    Args:
        state: Final attack state
        filepath: Output file path. If None, generates timestamp-based filename.
        model_info: Optional dict with model names (planner, generator, target, judge).

    Returns:
        The filepath where the HTML was saved.
    """
    assert osp.exists(osp.dirname(filepath)), f"Directory not found: {osp.dirname(filepath)}"

    # Load template
    template_content = _load_template("summary_template.html")
    template = Template(template_content)

    # Prepare data for template
    status_class = "status-success" if state.get('attack_is_success') else "status-failed"
    status_text = "SUCCESS" if state.get('attack_is_success') else "FAILED"
    scenario = state.get('scenario', {})

    # Build model info items if available
    model_info_html = ""
    if model_info:
        for role, model_name in model_info.items():
            if model_name:
                model_info_html += f"""            <div class="metadata-item">
                <span class="metadata-label">{role.title()} Model</span>
                <span class="metadata-value">{_escape_html(model_name)}</span>
            </div>
"""

    # Build attempts HTML
    attempts_html = _build_attempts_html(state.get('attack_history', []))

    # Build GA optimization history HTML (if available)
    ga_history_html = _build_ga_optimization_history_html(state.get('ga_optimization_history', []))

    # Render template
    html_content = template.substitute({
        'status_class': status_class,
        'status_text': status_text,
        'total_iterations': state.get('current_iteration', 1),
        'scenario_id': scenario.get('scenario_id', 'N/A'),
        'final_stage': state.get('current_stage', 'N/A'),
        'intention': _escape_html(state.get('intention', 'N/A')),
        'model_info_html': model_info_html,
        'attempts_html': attempts_html,
        'ga_history_html': ga_history_html,
    })

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML summary saved to: {filepath}")
    return filepath


def _build_attempts_html(attack_history: List[AttackState]) -> str:
    """Build HTML for all attack attempts.

    Args:
        attack_history: List of attack attempt records

    Returns:
        HTML string for all attempts
    """
    attempts_html = []
    for i, record in enumerate(attack_history, 1):
        attempts_html.append(_build_attempt_html(i, record))
    return '\n'.join(attempts_html)


def _build_attempt_html(attempt_num: int, record: AttackState) -> str:
    """Build HTML for a single attack attempt.

    Args:
        attempt_num: Attempt number (1-indexed)
        record: Attack attempt record

    Returns:
        HTML string for the attempt
    """
    attempt_class = "attempt-success" if record.get('attack_is_success') else "attempt-failed"
    success_badge = "✓ SUCCESS" if record.get('attack_is_success') else "✗ FAILED"
    badge_class = "badge-valid" if record.get('attack_is_success') else "badge-invalid"

    parts = [
        f'''        <div class="attempt {attempt_class}">
            <div class="attempt-header">
                <span>ATTEMPT {attempt_num} (Iteration {record.get("current_iteration")})</span>
                <span class="badge {badge_class}">{success_badge}</span>
            </div>'''
    ]

    # Add sections
    parts.append(_build_planner_section_html(attempt_num, record))
    parts.append(_build_subgraph_optimize_history_html(attempt_num, record))
    parts.append(_build_wrapped_prompt_section_html(attempt_num, record))
    parts.append(_build_target_response_section_html(attempt_num, record))
    parts.append(_build_judge_section_html(record))

    parts.append('        </div>')
    return '\n'.join(parts)


def _build_planner_section_html(attempt_num: int, record: AttackState) -> str:
    """Build HTML for the planner section.

    Args:
        attempt_num: Attempt number for unique IDs
        record: Attack attempt record

    Returns:
        HTML string for the planner section
    """
    planner_output = record.get("planner_output", {})
    planner_raw_output = planner_output.get('planner_raw_output', {})

    parse_success = planner_output.get('parse_success', True)
    parse_badge = f"badge-{'valid' if parse_success else 'invalid'}"
    parse_text = "✓ PARSED" if parse_success else "✗ PARSE FAILED"
    raw_output_id = f"planner_raw_{attempt_num}"

    parts = [
        '''            <div class="section">
                <div class="section-header section-planner">PLANNER</div>
                <div class="section-content">''',
        f'                    <div class="row"><span class="key">Stage:</span><span class="value">{planner_output.get("next_stage", "N/A")}</span></div>',
        f'                    <div class="row"><span class="key">Cell ID:</span><span class="value">{planner_output.get("cell_id", "N/A")}</span></div>',
        f'                    <div class="row"><span class="key">Parse Status:</span><span class="badge {parse_badge}">{parse_text}</span></div>'
    ]

    if planner_raw_output:
        parts.append(f'''
                    <button class="toggle" onclick="toggle('{raw_output_id}')">Toggle Raw Planner Output ({len(planner_raw_output)} chars)</button>
                    <div id="{raw_output_id}" class="collapsible">
                        <div class="code-block">{_escape_html(planner_raw_output)}</div>
                    </div>''')

    parts.append('                </div>\n            </div>')
    return '\n'.join(parts)


def _build_wrapped_prompt_section_html(attempt_num: int, record: AttackState) -> str:
    """Build HTML for the wrapped prompt section.

    Args:
        attempt_num: Attempt number for unique IDs
        record: Attack attempt record

    Returns:
        HTML string for the wrapped prompt section
    """
    wrapped_prompt = record.get("wrapped_prompt", "")
    prompt_id = f"prompt_{attempt_num}"

    return f'''            <div class="section">
                <div class="section-header section-generator">Wrapped Prompt</div>
                <div class="section-content">
                    <button class="toggle" onclick="toggle('{prompt_id}')">Toggle Full Prompt ({len(wrapped_prompt)} chars)</button>
                    <div id="{prompt_id}" class="collapsible">
                        <div class="code-block">{_escape_html(wrapped_prompt)}</div>
                    </div>
                </div>
            </div>'''


def _build_subgraph_optimize_history_html(attempt_num: int, record: AttackState) -> str:
    """Build HTML for the subgraph optimization history section.

    Args:
        attempt_num: Attempt number for unique IDs
        record: Attack attempt record

    Returns:
        HTML string for the subgraph optimization history section
    """
    optimize_history = record.get("subgraph_optimize_history", [])

    if not optimize_history:
        return ""

    history_id = f"subgraph_history_{attempt_num}"
    parts = [
        f'''            <div class="section">
                <div class="section-header section-generator">Subgraph Optimization History</div>
                <div class="section-content">
                    <div class="row"><span class="key">Iterations:</span><span class="value">{len(optimize_history)}</span></div>
                    <button class="toggle" onclick="toggle('{history_id}')">Toggle Optimization Details</button>
                    <div id="{history_id}" class="collapsible">'''
    ]

    for i, iteration in enumerate(optimize_history, 1):
        selected_dims = iteration.get("selected_dimensions", [])
        strategy_plan = iteration.get("strategy_plan", {})
        wrapped_prompt = iteration.get("wrapped_prompt", "")
        manip_check = iteration.get("manipulation_check", {})
        stage_alignment = iteration.get("stage_alignment", manip_check.get("stage_alignment", "N/A"))
        stage_leakage = iteration.get("stage_leakage", manip_check.get("stage_leakage", "N/A"))
        decision = manip_check.get("decision", "N/A")

        dims_display = ", ".join(selected_dims) if selected_dims else "None"
        plan_display = _escape_html(json.dumps(strategy_plan, ensure_ascii=False))

        parts.append(f'''
                        <div class="subgraph-iteration">
                            <div class="subgraph-iteration-header">Iteration {i}</div>
                            <div class="row"><span class="key">Selected Dimensions:</span><span class="value">{_escape_html(dims_display)}</span></div>
                            <div class="row"><span class="key">Stage Alignment:</span><span class="value">{stage_alignment}</span></div>
                            <div class="row"><span class="key">Stage Leakage:</span><span class="value">{stage_leakage}</span></div>
                            <div class="row"><span class="key">Manip Check Decision:</span><span class="value">{decision}</span></div>
                            <div class="row"><span class="key">Strategy Plan:</span><span class="value">{plan_display}</span></div>''')

        if wrapped_prompt:
            prompt_preview = _escape_html(wrapped_prompt)
            parts.append(f'''
                            <div class="row"><span class="key">Prompt Preview:</span></div>
                            <div class="code-block subgraph-prompt">{prompt_preview}</div>''')

        parts.append('                        </div>')

    parts.append('''
                    </div>
                </div>
            </div>''')

    return '\n'.join(parts)


def _build_target_response_section_html(attempt_num: int, record: AttackState) -> str:
    """Build HTML for the target response section.

    Args:
        attempt_num: Attempt number for unique IDs
        record: Attack attempt record

    Returns:
        HTML string for the target response section
    """
    target_response = record.get("target_response", "")
    response_id = f"response_{attempt_num}"

    return f'''            <div class="section">
                <div class="section-header section-target">TARGET AGENT</div>
                <div class="section-content">
                    <button class="toggle" onclick="toggle('{response_id}')">Toggle Full Response ({len(target_response)} chars)</button>
                    <div id="{response_id}" class="collapsible">
                        <div class="code-block">{_escape_html(target_response)}</div>
                    </div>
                </div>
            </div>'''


def _build_judge_section_html(record: AttackState) -> str:
    """Build HTML for the judge section.

    Args:
        record: Attack attempt record

    Returns:
        HTML string for the judge section
    """
    judge_result = record.get("judge_result", {})
    judgement_score = judge_result.get("judgement_score", "N/A")
    judgement_reason = judge_result.get("thereason", "")
    mechanism_gain = judge_result.get("mechanism_gain", "N/A")
    stage_alignment = judge_result.get("stage_alignment", record.get("stage_alignment", "N/A"))
    stage_leakage = judge_result.get("stage_leakage", record.get("stage_leakage", "N/A"))
    manip_check_pass = judge_result.get("manip_check_pass", record.get("manip_check_pass", "N/A"))
    raw_response = judge_result.get("judgment_raw_response", "")

    # Determine badge style based on score (1-4, higher is better)
    score_badge_class = "badge-valid"
    if isinstance(judgement_score, int):
        if judgement_score >= 4:
            score_badge_class = "badge-valid"
        elif judgement_score >= 3:
            score_badge_class = "badge-warning"
        else:
            score_badge_class = "badge-invalid"

    raw_response_id = f"judge_raw_{record.get('current_iteration', 'unknown')}"

    parts = [
        '''            <div class="section">
                <div class="section-header section-judge">JUDGE</div>
                <div class="section-content">
                    <div class="row">
                        <div class="col">
                            <span class="key">Judgement Score:</span>
                            <span class="badge ''' + score_badge_class + '''">''' + str(judgement_score) + '''</span>
                        </div>
                    </div>'''
    ]

    parts.append(f'                    <div><span class="key">Mechanism Gain:</span><span class="value">{mechanism_gain}</span></div>')
    parts.append(f'                    <div><span class="key">Stage Alignment:</span><span class="value">{stage_alignment}</span></div>')
    parts.append(f'                    <div><span class="key">Stage Leakage:</span><span class="value">{stage_leakage}</span></div>')
    parts.append(f'                    <div><span class="key">Manip Check Pass:</span><span class="value">{manip_check_pass}</span></div>')
    if judgement_reason:
        reason_escaped = _escape_html(str(judgement_reason)[:300])
        if len(str(judgement_reason)) > 300:
            reason_escaped += "..."
        parts.append(f'                    <div><span class="key">Reason:</span><span class="value">{reason_escaped}</span></div>')

    # Add collapsible raw response if available
    if raw_response:
        parts.append(f'''
                    <button class="toggle" onclick="toggle('{raw_response_id}')">Toggle Raw Judge Response ({len(raw_response)} chars)</button>
                    <div id="{raw_response_id}" class="collapsible">
                        <div class="code-block">{_escape_html(raw_response)}</div>
                    </div>''')

    parts.append('                </div>\n            </div>')
    return '\n'.join(parts)


def _build_ga_optimization_history_html(ga_history: List[Dict[str, Any]]) -> str:
    """Build HTML for the GA optimization history section.

    Args:
        ga_history: List of generation records from GA optimization

    Returns:
        HTML string for the GA optimization history section
    """
    if not ga_history:
        return ""

    total_generations = len(ga_history)
    history_id = "ga_optimization_history"

    parts = [
        f'''        <div class="section">
            <div class="section-header section-planner">GA Optimization History</div>
            <div class="section-content">
                <div class="row"><span class="key">Total Generations:</span><span class="value">{total_generations}</span></div>'''
    ]

    for gen_data in ga_history:
        generation = gen_data.get("generation", 0)
        avg_fitness = gen_data.get("avg_fitness", 0.0)
        best_fitness = gen_data.get("best_fitness", 0.0)
        worst_fitness = gen_data.get("worst_fitness", 0.0)
        population = gen_data.get("population", [])

        gen_id = f"gen_{generation}"
        gen_section_id = f"gen_section_{generation}"

        parts.append(f'''
                <div class="ga-generation">
                    <div class="ga-generation-header">
                        <button class="toggle" onclick="toggle('{gen_section_id}')">
                            Toggle Generation {generation + 1}
                        </button>
                        <span class="ga-generation-stats">
                            Avg: {avg_fitness:.2f} | Best: {best_fitness:.2f} | Worst: {worst_fitness:.2f}
                        </span>
                    </div>
                    <div id="{gen_section_id}" class="collapsible">
                        <div class="ga-population">''')

        # Add each individual in the population
        for idx, individual in enumerate(population):
            fitness = individual.get("fitness", "N/A")
            score = individual.get("score", "N/A")
            mechanism_gain = individual.get("mechanism_gain", "N/A")
            chromosome_data = individual.get("chromosome", {})
            final_result = individual.get("final_result")

            # Format stage selections
            stage_selections = chromosome_data.get("stage_selections", {})
            selections_display = []
            for stage, dims in stage_selections.items():
                dims_str = ", ".join(dims) if dims else "None"
                selections_display.append(f"{stage}: [{dims_str}]")

            chromosome_display = " | ".join(selections_display) if selections_display else "N/A"

            # Style the best individual
            is_best = (fitness == best_fitness)
            row_class = "ga-individual-best" if is_best else "ga-individual"
            final_result_id = f"final_result_gen{generation}_idx{idx}"

            parts.append(f'''
                            <div class="{row_class}">
                                <div class="ga-individual-header">
                                    <span class="ga-individual-index">Individual {idx}</span>
                                    <span class="badge {'badge-valid' if is_best and fitness >= 4 else 'badge-warning'}">
                                        Fitness: {fitness}
                                    </span>
                                </div>
                                <div class="row"><span class="key">Score:</span><span class="value">{score}</span></div>
                                <div class="row"><span class="key">Mechanism Gain:</span><span class="value">{mechanism_gain}</span></div>
                                <div class="row"><span class="key">Stage Selections:</span><span class="value">{_escape_html(chromosome_display)}</span></div>''')

            # Add final_result section if available
            if final_result:
                final_status = "Success" if final_result.get("attack_is_success") else "Failed"
                status_class = "badge-valid" if final_result.get("attack_is_success") else "badge-invalid"
                parts.append(f'''
                                <div class="row" style="margin-top: 10px;">
                                    <span class="key">Full Attack Result:</span>
                                    <button class="toggle" onclick="toggle('{final_result_id}')">Show Details</button>
                                </div>
                                <div id="{final_result_id}" class="collapsible">
                                    <div class="row"><span class="key">Status:</span><span class="value"><span class="badge {status_class}">{final_status}</span></span></div>
                                    <div class="row"><span class="key">Iterations:</span><span class="value">{final_result.get("current_iteration", "N/A")}</span></div>
                                    <div class="row"><span class="key">Final Stage:</span><span class="value">{final_result.get("current_stage", "N/A")}</span></div>''')

                # Add wrapped_prompt if available
                wrapped_prompt = final_result.get("wrapped_prompt", "")
                if wrapped_prompt:
                    prompt_detail_id = f"{final_result_id}_prompt"
                    parts.append(f'''
                                    <div class="row" style="margin-top: 10px;">
                                        <span class="key">Wrapped Prompt:</span>
                                        <button class="toggle" onclick="toggle('{prompt_detail_id}')">Show ({len(wrapped_prompt)} chars)</button>
                                    </div>
                                    <div id="{prompt_detail_id}" class="collapsible">
                                        <div class="code-block">{_escape_html(wrapped_prompt)}</div>
                                    </div>''')

                # Add target_response if available
                target_response = final_result.get("target_response", "")
                if target_response:
                    response_detail_id = f"{final_result_id}_response"
                    parts.append(f'''
                                    <div class="row" style="margin-top: 10px;">
                                        <span class="key">Target Response:</span>
                                        <button class="toggle" onclick="toggle('{response_detail_id}')">Show ({len(target_response)} chars)</button>
                                    </div>
                                    <div id="{response_detail_id}" class="collapsible">
                                        <div class="code-block">{_escape_html(target_response)}</div>
                                    </div>''')

                parts.append('''
                                </div>''')

            parts.append('''
                            </div>''')

        parts.append('''
                        </div>
                    </div>
                </div>''')

    parts.append('''
            </div>
        </div>''')

    return '\n'.join(parts)


def generate_dashboard(results: List[Dict[str, Any]], results_dir: str, model_name: str) -> str:
    """Generate an HTML dashboard with visualizations of batch test results.

    Args:
        results: List of test results
        results_dir: Directory to save dashboard
        model_name: Name of the tested model

    Returns:
        Path to the dashboard HTML file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name)).strip("_")
    if not model_tag:
        model_tag = "unknown_model"
    dashboard_path = Path(results_dir) / f"dashboard_{model_tag}_{timestamp}.html"

    # Load template
    template_content = _load_template("dashboard_template.html")
    template = Template(template_content)

    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    success_rate = successful / total * 100 if total > 0 else 0
    avg_iterations = sum(r.get("iterations", 0) for r in results) / total if total > 0 else 0

    # Score distribution
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in results:
        score_counts[r["score"]] = score_counts.get(r["score"], 0) + 1

    # Build chart HTML
    score_distribution_html = _build_score_distribution_html(score_counts, total)
    success_failure_html = _build_success_failure_html(successful, failed, success_rate)
    results_table_rows = _build_results_table_rows(results)

    # Render template
    html_content = template.substitute({
        'model_name': model_name,
        'timestamp': timestamp,
        'total_samples': total,
        'successful_attacks': successful,
        'failed_attacks': failed,
        'success_rate': f"{success_rate:.1f}",
        'avg_iterations': f"{avg_iterations:.2f}",
        'score_distribution_html': score_distribution_html,
        'success_failure_html': success_failure_html,
        'results_table_rows': results_table_rows,
    })

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Dashboard HTML saved to: {dashboard_path}")
    return str(dashboard_path)


def _build_score_distribution_html(score_counts: Dict[int, int], total: int) -> str:
    """Build HTML for score distribution chart."""
    parts = []
    for score in [1, 2, 3, 4, 5]:
        count = score_counts.get(score, 0)
        percentage = count / total * 100 if total > 0 else 0
        parts.append(
            f'''                    <div class="bar-row">
                        <div class="bar-label">Score {score}</div>
                        <div class="bar-container">
                            <div class="bar score-{score}" style="width: {percentage:.1f}%">
                                <span class="bar-value">{count} ({percentage:.1f}%)</span>
                            </div>
                        </div>
                        <div class="bar-note">n={count}</div>
                    </div>'''
        )
    return "\n".join(parts)


def _build_success_failure_html(successful: int, failed: int, success_rate: float) -> str:
    """Build HTML for success vs failure chart.

    Args:
        successful: Number of successful attacks
        failed: Number of failed attacks
        success_rate: Success rate percentage

    Returns:
        HTML string for the success vs failure chart
    """
    return f'''                    <div class="bar-row">
                        <div class="bar-label">Success</div>
                        <div class="bar-container">
                            <div class="bar success" style="width: {success_rate:.1f}%">
                                <span class="bar-value">{successful} ({success_rate:.1f}%)</span>
                            </div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">Failed</div>
                        <div class="bar-container">
                            <div class="bar failed" style="width: {100-success_rate:.1f}%">
                                <span class="bar-value">{failed} ({100-success_rate:.1f}%)</span>
                            </div>
                        </div>
                    </div>'''


def _build_results_table_rows(results: List[Dict[str, Any]]) -> str:
    """Build HTML table rows for results.

    Args:
        results: List of test results

    Returns:
        HTML string for all table rows
    """
    rows = []
    for r in results:
        status_class = "success" if r["success"] else "failed"
        status_text = "SUCCESS" if r["success"] else "FAILED"
        html_link = f'<a href="{r["html_file"]}" target="_blank">View</a>' if r.get("html_file") else "-"
        json_link = f'<a href="{r["json_file"]}" target="_blank">JSON</a>' if r.get("json_file") else "-"
        error_text = f'<br><span class="error-text">{_escape_html(r["error"])}</span>' if r.get("error") else ""
        prompt_text = _escape_html(r['prompt'][:100] + ('...' if len(r['prompt']) > 100 else ''))
        cell_id = _escape_html(r.get('cell_id', 'n/a'))
        design_text = f"[{cell_id}]"

        rows.append(f'''        <tr class="{status_class}">
            <td>{r['index']}</td>
            <td class="prompt-cell">{design_text}<br>{prompt_text}{error_text}</td>
            <td class="score-cell">{r['score']}</td>
            <td class="status-cell">{status_text}</td>
            <td class="iter-cell">{r['iterations']}</td>
            <td class="links-cell">{html_link} | {json_link}</td>
        </tr>''')
    return '\n'.join(rows)

# ============================================================================
# Utilities for JSON Saving
# ============================================================================

def save_summary_json(
    state: AttackState,
    filepath: Optional[str] = None,
    model_info: Optional[Dict[str, str]] = None,
) -> str:
    """Save attack workflow summary as JSON.

    Args:
        state: Final attack state
        filepath: Output file path. If None, generates timestamp-based filename.
        model_info: Optional dict with model names (planner, generator, target, judge).

    Returns:
        The filepath where the JSON was saved.
    """
    assert osp.exists(osp.dirname(filepath)), f"Directory not found: {osp.dirname(filepath)}"

    output_data = {
        "metadata": {
            "intention": state.get('intention'),
            "scenario_id": state.get('scenario', {}).get('scenario_id'),
            "scenario_info": state.get('scenario', {}).get('scenario_info'),
            "final_status": "success" if state.get('attack_is_success') else "failed",
            "total_iterations": state.get('current_iteration', 1),
            "final_stage": state.get('current_stage'),
            "models": model_info or {},
        },
        "history": state.get('attack_history', []),
        "ga_optimization_history": state.get('ga_optimization_history', []),
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Summary saved to: {filepath}")
    return filepath


def save_batch_summary(results: List[Dict[str, Any]], results_dir: str) -> str:
    """Save batch test results to JSON.

    Args:
        results: List of test results
        results_dir: Directory to save results

    Returns:
        Path to the saved summary file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(results_dir) / f"batch_summary_{timestamp}.json"

    total = len(results)
    stage_metrics = compute_stage_metrics(results)

    summary = {
        "timestamp": timestamp,
        "total_samples": total,
        "successful_attacks": sum(1 for r in results if r["success"]),
        "failed_attacks": sum(1 for r in results if not r["success"]),
        "success_rate": sum(1 for r in results if r["success"]) / total * 100 if total > 0 else 0,
        "average_iterations": sum(r["iterations"] for r in results) / total if total > 0 else 0,
        "stage_metrics": stage_metrics,
        "results": results
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Batch summary JSON saved to: {summary_path}")
    return str(summary_path)
