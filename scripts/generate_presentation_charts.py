from pathlib import Path
import csv


def svg_wrap(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<style>
  text {{ font-family: "Segoe UI", Arial, sans-serif; fill: #1f2937; }}
  .title {{ font-size: 26px; font-weight: 700; }}
  .axis-title {{ font-size: 18px; font-weight: 600; }}
  .tick {{ font-size: 14px; fill: #374151; }}
  .label {{ font-size: 13px; font-weight: 600; }}
  .legend-title {{ font-size: 15px; font-weight: 700; }}
  .legend-text {{ font-size: 14px; }}
</style>
{body}
</svg>
"""


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")


def chart_cumulative_svg(models, colors, cumulative):
    width, height = 1200, 760
    left, right, top, bottom = 90, 320, 95, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    stages = [1, 2, 3, 4]

    def x_of(stage):
        return left + (stage - 1) * (plot_w / 3)

    def y_of(val_percent):
        return top + (100 - val_percent) / 100 * plot_h

    body = []
    body.append('<text class="title" x="600" y="48" text-anchor="middle">Cumulative Success Rate by Stage (C_k)</text>')
    body.append('<text class="tick" x="600" y="72" text-anchor="middle">N = 20 per model</text>')

    for t in range(0, 101, 20):
        y = y_of(t)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text class="tick" x="{left - 10}" y="{y + 5:.1f}" text-anchor="end">{t}%</text>')

    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>')

    for stage in stages:
        x = x_of(stage)
        body.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 6}" stroke="#111827" stroke-width="1.5"/>')
        body.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle">Stage {stage}</text>')

    body.append(f'<text class="axis-title" x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle">Stage k</text>')
    body.append(f'<text class="axis-title" x="28" y="{top + plot_h / 2:.1f}" transform="rotate(-90,28,{top + plot_h / 2:.1f})" text-anchor="middle">C_k (%)</text>')

    for model in models:
        points = []
        for s, v in enumerate(cumulative[model], start=1):
            points.append(f"{x_of(s):.1f},{y_of(v * 100):.1f}")
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model]}" stroke-width="3.5"/>')

        for s, v in enumerate(cumulative[model], start=1):
            x = x_of(s)
            y = y_of(v * 100)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[model]}" stroke="#ffffff" stroke-width="1.5"/>')
            body.append(f'<text class="label" x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" fill="{colors[model]}">{v * 100:.0f}%</text>')

    lx = left + plot_w + 30
    ly = top + 70
    body.append(f'<text class="legend-title" x="{lx}" y="{ly - 20}">Model</text>')
    for i, model in enumerate(models):
        y = ly + i * 32
        body.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="{colors[model]}" stroke-width="4"/>')
        body.append(f'<circle cx="{lx + 17}" cy="{y}" r="5" fill="{colors[model]}" stroke="#fff" stroke-width="1"/>')
        body.append(f'<text class="legend-text" x="{lx + 45}" y="{y + 5}">{model}</text>')

    return svg_wrap(width, height, "\n".join(body))


def chart_delta_svg(models, colors, delta):
    width, height = 1250, 780
    left, right, top, bottom = 90, 360, 95, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    stages = [1, 2, 3, 4]
    max_y = 60
    bar_w = 34
    group_w = plot_w / 4
    offsets = {"GPT-5.2-chat": -40, "DeepSeek-Reasoner": 0, "Qwen3-8B": 40}

    def x_center(stage):
        return left + (stage - 0.5) * group_w

    def y_of(val_percent):
        return top + (max_y - val_percent) / max_y * plot_h

    body = []
    body.append('<text class="title" x="620" y="48" text-anchor="middle">Stage Incremental Gain (Delta_k)</text>')
    body.append('<text class="tick" x="620" y="72" text-anchor="middle">Green marker: Delta_k &gt; 2% | Red marker: Delta_k &lt;= 2%</text>')

    for t in range(0, max_y + 1, 10):
        y = y_of(t)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text class="tick" x="{left - 10}" y="{y + 5:.1f}" text-anchor="end">{t}%</text>')

    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="2"/>')

    for stage in stages:
        xc = x_center(stage)
        body.append(f'<text class="tick" x="{xc:.1f}" y="{top + plot_h + 28}" text-anchor="middle">Stage {stage}</text>')
        body.append(f'<line x1="{xc:.1f}" y1="{top + plot_h}" x2="{xc:.1f}" y2="{top + plot_h + 6}" stroke="#111827" stroke-width="1.5"/>')

    body.append(f'<text class="axis-title" x="{left + plot_w / 2:.1f}" y="{height - 30}" text-anchor="middle">Stage k</text>')
    body.append(f'<text class="axis-title" x="28" y="{top + plot_h / 2:.1f}" transform="rotate(-90,28,{top + plot_h / 2:.1f})" text-anchor="middle">Delta_k (%)</text>')

    for model in models:
        for stage, v in enumerate(delta[model], start=1):
            vp = v * 100
            xc = x_center(stage) + offsets[model]
            x = xc - bar_w / 2
            y = y_of(vp)
            h = top + plot_h - y
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{colors[model]}" opacity="0.9"/>')

            mark_color = "#2ca02c" if vp > 2 else "#d62728"
            my = y - 10
            body.append(f'<circle cx="{xc:.1f}" cy="{my:.1f}" r="6" fill="{mark_color}" stroke="#fff" stroke-width="1.2"/>')
            body.append(f'<text class="label" x="{xc:.1f}" y="{my - 12:.1f}" text-anchor="middle" fill="{colors[model]}">{vp:.0f}%</text>')

    lx = left + plot_w + 25
    ly = top + 55
    body.append(f'<text class="legend-title" x="{lx}" y="{ly - 18}">Model</text>')
    for i, model in enumerate(models):
        y = ly + i * 32
        body.append(f'<rect x="{lx}" y="{y - 12}" width="22" height="14" fill="{colors[model]}"/>')
        body.append(f'<text class="legend-text" x="{lx + 32}" y="{y}">{model}</text>')

    sy = ly + 140
    body.append(f'<text class="legend-title" x="{lx}" y="{sy - 18}">Gain Status</text>')
    body.append(f'<circle cx="{lx + 10}" cy="{sy}" r="6" fill="#2ca02c" stroke="#fff" stroke-width="1"/>')
    body.append(f'<text class="legend-text" x="{lx + 24}" y="{sy + 5}">Delta_k &gt; 2% (Meaningful gain)</text>')
    body.append(f'<circle cx="{lx + 10}" cy="{sy + 28}" r="6" fill="#d62728" stroke="#fff" stroke-width="1"/>')
    body.append(f'<text class="legend-text" x="{lx + 24}" y="{sy + 33}">Delta_k &lt;= 2% (Plateau)</text>')

    return svg_wrap(width, height, "\n".join(body))


def chart_convergence_svg(models, colors, convergence):
    width, height = 1280, 760
    left, right, top, bottom = 70, 70, 95, 90
    panel_gap = 70
    panel_w = (width - left - right - panel_gap) / 2
    panel_h = height - top - bottom

    ratio_x0 = left
    delta_x0 = left + panel_w + panel_gap

    def y_ratio(v):
        return top + (100 - v) / 20 * panel_h

    def y_delta(v):
        return top + (6 - v) / 6 * panel_h

    body = []
    body.append('<text class="title" x="640" y="48" text-anchor="middle">Convergence Diagnostic (C3/C4 and Delta_4)</text>')

    for t in [80, 85, 90, 95, 100]:
        y = y_ratio(t)
        body.append(f'<line x1="{ratio_x0}" y1="{y:.1f}" x2="{ratio_x0 + panel_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text class="tick" x="{ratio_x0 - 8}" y="{y + 5:.1f}" text-anchor="end">{t}%</text>')

    body.append(f'<line x1="{ratio_x0}" y1="{top}" x2="{ratio_x0}" y2="{top + panel_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<line x1="{ratio_x0}" y1="{top + panel_h}" x2="{ratio_x0 + panel_w}" y2="{top + panel_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<text class="axis-title" x="{ratio_x0 + panel_w / 2:.1f}" y="{height - 28}" text-anchor="middle">Model</text>')
    body.append(f'<text class="axis-title" x="{ratio_x0 + panel_w / 2:.1f}" y="{top - 20}" text-anchor="middle">Convergence Ratio (C3/C4)</text>')

    body.append(f'<line x1="{ratio_x0}" y1="{y_ratio(95):.1f}" x2="{ratio_x0 + panel_w}" y2="{y_ratio(95):.1f}" stroke="#374151" stroke-width="2" stroke-dasharray="7 5"/>')
    body.append(f'<text class="tick" x="{ratio_x0 + panel_w - 6}" y="{y_ratio(95) - 8:.1f}" text-anchor="end">Threshold 95%</text>')

    bw = panel_w / 6
    for i, model in enumerate(models):
        x = ratio_x0 + (i * 2 + 1) * bw
        v = convergence[model]["c3_c4"] * 100
        y = y_ratio(v)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{top + panel_h - y:.1f}" fill="{colors[model]}" opacity="0.9"/>')
        body.append(f'<text class="tick" x="{x + bw / 2:.1f}" y="{top + panel_h + 24}" text-anchor="middle">{model}</text>')
        body.append(f'<text class="label" x="{x + bw / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle">{v:.1f}%</text>')

    for t in [0, 1, 2, 3, 4, 5, 6]:
        y = y_delta(t)
        body.append(f'<line x1="{delta_x0}" y1="{y:.1f}" x2="{delta_x0 + panel_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text class="tick" x="{delta_x0 - 8}" y="{y + 5:.1f}" text-anchor="end">{t}%</text>')

    body.append(f'<line x1="{delta_x0}" y1="{top}" x2="{delta_x0}" y2="{top + panel_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<line x1="{delta_x0}" y1="{top + panel_h}" x2="{delta_x0 + panel_w}" y2="{top + panel_h}" stroke="#111827" stroke-width="2"/>')
    body.append(f'<text class="axis-title" x="{delta_x0 + panel_w / 2:.1f}" y="{height - 28}" text-anchor="middle">Model</text>')
    body.append(f'<text class="axis-title" x="{delta_x0 + panel_w / 2:.1f}" y="{top - 20}" text-anchor="middle">Last-stage Gain (Delta_4)</text>')

    body.append(f'<line x1="{delta_x0}" y1="{y_delta(2):.1f}" x2="{delta_x0 + panel_w}" y2="{y_delta(2):.1f}" stroke="#374151" stroke-width="2" stroke-dasharray="7 5"/>')
    body.append(f'<text class="tick" x="{delta_x0 + panel_w - 6}" y="{y_delta(2) - 8:.1f}" text-anchor="end">Threshold 2%</text>')

    for i, model in enumerate(models):
        x = delta_x0 + (i * 2 + 1) * bw
        v = convergence[model]["delta4"] * 100
        y = y_delta(v)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{top + panel_h - y:.1f}" fill="{colors[model]}" opacity="0.9"/>')
        body.append(f'<text class="tick" x="{x + bw / 2:.1f}" y="{top + panel_h + 24}" text-anchor="middle">{model}</text>')
        body.append(f'<text class="label" x="{x + bw / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle">{v:.0f}%</text>')
        body.append(f'<text class="tick" x="{x + bw / 2:.1f}" y="{top + panel_h + 44}" text-anchor="middle">{convergence[model]["decision"]}</text>')

    return svg_wrap(width, height, "\n".join(body))


def main() -> None:
    output_dir = Path("presentation_charts")
    output_dir.mkdir(exist_ok=True)

    models = ["GPT-5.2-chat", "DeepSeek-Reasoner", "Qwen3-8B"]
    colors = {
        "GPT-5.2-chat": "#1f77b4",
        "DeepSeek-Reasoner": "#ff7f0e",
        "Qwen3-8B": "#2ca02c",
    }
    cumulative = {
        "GPT-5.2-chat": [0.18, 0.37, 0.54, 0.61],
        "DeepSeek-Reasoner": [0.25, 0.55, 0.82, 0.90],
        "Qwen3-8B": [0.40, 0.75, 0.93, 0.95],
    }
    delta = {
        "GPT-5.2-chat": [0.18, 0.19, 0.17, 0.07],
        "DeepSeek-Reasoner": [0.25, 0.30, 0.27, 0.08],
        "Qwen3-8B": [0.40, 0.35, 0.18, 0.02],
    }
    convergence = {
        "GPT-5.2-chat": {"c3_c4": 0.885, "delta4": 0.07, "decision": "No"},
        "DeepSeek-Reasoner": {"c3_c4": 0.911, "delta4": 0.08, "decision": "No"},
        "Qwen3-8B": {"c3_c4": 0.979, "delta4": 0.02, "decision": "Yes"},
    }

    c1 = output_dir / "chart_cumulative_success_rate.svg"
    c2 = output_dir / "chart_stage_incremental_gain.svg"
    c3 = output_dir / "chart_convergence.svg"
    save_text(c1, chart_cumulative_svg(models, colors, cumulative))
    save_text(c2, chart_delta_svg(models, colors, delta))
    save_text(c3, chart_convergence_svg(models, colors, convergence))

    csv_path = output_dir / "chart_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "C1", "C2", "C3", "C4", "Delta1", "Delta2", "Delta3", "Delta4", "C3_C4", "Convergence"])
        for model in models:
            writer.writerow([
                model,
                *[f"{v:.2f}" for v in cumulative[model]],
                *[f"{v:.2f}" for v in delta[model]],
                f"{convergence[model]['c3_c4']:.3f}",
                convergence[model]["decision"],
            ])

    print(f"Generated: {c1}")
    print(f"Generated: {c2}")
    print(f"Generated: {c3}")
    print(f"Generated: {csv_path}")


if __name__ == "__main__":
    main()
