#!/usr/bin/env python3
"""
论文检索系统 Web 界面 (Gradio)
复用 paper_search.py 的核心逻辑
"""

import argparse
import sys
from pathlib import Path

import gradio as gr

# 确保可以 import 同目录的 paper_search
sys.path.insert(0, str(Path(__file__).parent))
import paper_search as ps

# ---------------------------------------------------------------------------
# 全局状态（启动时通过命令行参数设置）
# ---------------------------------------------------------------------------
OAI_CLIENT = None
COLLECTION = None
DATA_DIR = None
EMBEDDING_MODEL = None

VENUE_CHOICES = ["All", "ICLR", "NeurIPS", "ICML", "ACL", "EMNLP"]
YEAR_CHOICES = ["All", "2024", "2025"]
CHAT_MODEL_CHOICES = [
    "aws/gpt-5.4",
    "bh/claude-sonnet-4-6",
    "bh/claude-opus-4-6",
    "bh/gemini-2.5-flash",
]


def init_globals(api_key, base_url, model, data):
    global OAI_CLIENT, COLLECTION, DATA_DIR, EMBEDDING_MODEL
    DATA_DIR = Path(data)
    EMBEDDING_MODEL = model
    OAI_CLIENT = ps.get_openai_client(api_key, base_url)
    COLLECTION = ps.get_collection(DATA_DIR)
    count = COLLECTION.count()
    print(f"索引已加载: {count} 篇论文")
    if count == 0:
        print("[警告] 索引为空，请先运行 build 命令")


# ---------------------------------------------------------------------------
# Search 逻辑
# ---------------------------------------------------------------------------

def do_search(query, venue, year, top_k, live):
    if not query.strip():
        return [], "请输入检索关键词"

    venue_filter = None if venue == "All" else venue
    year_filter = None if year == "All" else int(year)
    top_k = int(top_k)

    all_results = []
    seen_ids = set()

    # 本地检索
    if COLLECTION.count() > 0:
        local = ps.search_local(
            COLLECTION, OAI_CLIENT, query, top_k,
            venue_filter, year_filter, model=EMBEDDING_MODEL
        )
        for p in local:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    # 实时检索
    if live:
        live_results = ps.search_live(query, top_k, venue_filter, year_filter)
        for p in live_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    if not all_results:
        return [], "未找到相关论文"

    results = all_results[:top_k]

    # 构建表格数据
    table_data = []
    for i, p in enumerate(results, 1):
        score = f"{p['score']:.2f}" if p["score"] > 0 else "-"
        table_data.append([
            i, score, p["source"],
            p.get("venue", ""), str(p.get("year", "")),
            p.get("title", ""),
        ])

    # 构建详情文本（所有论文的摘要）
    details_parts = []
    for i, p in enumerate(results, 1):
        arxiv_link = f"[ArXiv](https://arxiv.org/abs/{p['arxiv_id']})" if p.get("arxiv_id") else ""
        pdf_link = f"[PDF]({p['pdf_url']})" if p.get("pdf_url") else ""
        links = " | ".join(filter(None, [arxiv_link, pdf_link]))
        details_parts.append(
            f"### [{i}] {p['title']}\n"
            f"**作者:** {p.get('authors', '')}\n\n"
            f"**会议:** {p.get('venue', '')} {p.get('year', '')}  {links}\n\n"
            f"**摘要:** {p.get('abstract', '(无)')}\n\n---"
        )

    details_md = "\n\n".join(details_parts)
    return table_data, details_md


# ---------------------------------------------------------------------------
# Ask 逻辑
# ---------------------------------------------------------------------------

def do_ask(question, venue, year, top_k, live, chat_model):
    if not question.strip():
        return "请输入研究问题", "", []

    venue_filter = None if venue == "All" else venue
    year_filter = None if year == "All" else int(year)
    top_k = int(top_k)

    all_results = []
    seen_ids = set()

    if COLLECTION.count() > 0:
        local = ps.search_local(
            COLLECTION, OAI_CLIENT, question, top_k,
            venue_filter, year_filter, model=EMBEDDING_MODEL
        )
        for p in local:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    if live:
        live_results = ps.search_live(question, top_k, venue_filter, year_filter)
        for p in live_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    if not all_results:
        return "未找到相关论文，无法回答", "", []

    papers = all_results[:top_k]
    context = ps.build_context_from_papers(papers)
    answer = ps.ask_llm(OAI_CLIENT, question, context, chat_model)

    # 引用论文列表
    refs_parts = []
    for i, p in enumerate(papers, 1):
        pdf_link = f" [PDF]({p['pdf_url']})" if p.get("pdf_url") else ""
        refs_parts.append(
            f"[{i}] {p['title']}  "
            f"*({p.get('venue', '')} {p.get('year', '')})*{pdf_link}"
        )
    refs_md = "\n\n".join(refs_parts)

    return answer, refs_md, papers


def do_followup(followup_question, chat_model, papers):
    if not followup_question.strip():
        return "请输入追问内容"
    if not papers:
        return "没有论文上下文，请先提问"

    context = ps.build_context_from_papers(papers)
    answer = ps.ask_llm(OAI_CLIENT, followup_question, context, chat_model)
    return answer


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Minerva 论文检索系统") as app:
        gr.Markdown("# Minerva 论文检索系统\n"
                     "基于 NeurIPS / ICML / ICLR / ACL / EMNLP 2024-2025 论文库")

        # ===== Tab 1: Search =====
        with gr.Tab("论文检索"):
            with gr.Row():
                search_input = gr.Textbox(
                    label="检索关键词",
                    placeholder="例如：jailbreak attacks on large language models",
                    scale=4,
                )
                search_btn = gr.Button("检索", variant="primary", scale=1)

            with gr.Row():
                search_venue = gr.Dropdown(VENUE_CHOICES, value="All", label="会议")
                search_year = gr.Dropdown(YEAR_CHOICES, value="All", label="年份")
                search_top = gr.Slider(5, 50, value=20, step=5, label="返回数量")
                search_live = gr.Checkbox(label="实时补充", value=False)

            search_table = gr.Dataframe(
                headers=["#", "Score", "Source", "Venue", "Year", "Title"],
                datatype=["number", "str", "str", "str", "str", "str"],
                label="检索结果",
                interactive=False,
                wrap=True,
            )

            search_details = gr.Markdown(label="论文详情")

            search_btn.click(
                fn=do_search,
                inputs=[search_input, search_venue, search_year, search_top, search_live],
                outputs=[search_table, search_details],
            )
            search_input.submit(
                fn=do_search,
                inputs=[search_input, search_venue, search_year, search_top, search_live],
                outputs=[search_table, search_details],
            )

        # ===== Tab 2: Ask =====
        with gr.Tab("研究问答"):
            papers_state = gr.State([])

            with gr.Row():
                ask_input = gr.Textbox(
                    label="研究问题",
                    placeholder="例如：multi-turn jailbreak attack 的 SOTA 方法是什么？ASR 大概多少？",
                    scale=4,
                )
                ask_btn = gr.Button("提问", variant="primary", scale=1)

            with gr.Row():
                ask_venue = gr.Dropdown(VENUE_CHOICES, value="All", label="会议")
                ask_year = gr.Dropdown(YEAR_CHOICES, value="All", label="年份")
                ask_top = gr.Slider(10, 50, value=30, step=5, label="参考论文数")
                ask_live = gr.Checkbox(label="实时补充", value=False)
                ask_chat_model = gr.Dropdown(
                    CHAT_MODEL_CHOICES, value=CHAT_MODEL_CHOICES[0], label="Chat 模型"
                )

            answer_output = gr.Markdown(label="回答")

            with gr.Accordion("参考论文列表", open=False):
                refs_output = gr.Markdown()

            ask_btn.click(
                fn=do_ask,
                inputs=[ask_input, ask_venue, ask_year, ask_top, ask_live, ask_chat_model],
                outputs=[answer_output, refs_output, papers_state],
            )
            ask_input.submit(
                fn=do_ask,
                inputs=[ask_input, ask_venue, ask_year, ask_top, ask_live, ask_chat_model],
                outputs=[answer_output, refs_output, papers_state],
            )

            gr.Markdown("---\n### 追问（基于同一批论文）")
            with gr.Row():
                followup_input = gr.Textbox(
                    label="追问",
                    placeholder="基于上面检索到的论文继续追问...",
                    scale=4,
                )
                followup_btn = gr.Button("追问", scale=1)

            followup_output = gr.Markdown(label="追问回答")

            followup_btn.click(
                fn=do_followup,
                inputs=[followup_input, ask_chat_model, papers_state],
                outputs=[followup_output],
            )
            followup_input.submit(
                fn=do_followup,
                inputs=[followup_input, ask_chat_model, papers_state],
                outputs=[followup_output],
            )

    return app


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="论文检索系统 Web 界面")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="OpenAI API base URL")
    parser.add_argument("--model", default=ps.DEFAULT_EMBEDDING_MODEL, help="Embedding 模型名")
    parser.add_argument("--data", "-d", default="papers", help="数据目录 (默认: papers/)")
    parser.add_argument("--port", type=int, default=7860, help="端口 (默认: 7860)")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    args = parser.parse_args()

    init_globals(args.api_key, args.base_url, args.model, args.data)

    app = build_ui()
    app.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
