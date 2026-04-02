#!/usr/bin/env python3
"""
论文语义检索工具
基于 ChromaDB 向量索引 + OpenAI Embedding + Semantic Scholar 实时补充
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
import requests
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 20
COLLECTION_NAME = "papers"

S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,abstract,venue,year,externalIds,openAccessPdf"

DEFAULT_CHAT_MODEL = "aws/gpt-5.4"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_openai_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    url = base_url or os.environ.get("OPENAI_BASE_URL")
    if not key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量或使用 --api-key 参数")
        sys.exit(1)
    kwargs = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def embed_texts(client: OpenAI, texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL,
                max_retries: int = 5) -> list[list[float]]:
    """批量生成 embedding，带重试."""
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                all_embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"\n  Embedding 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    print(f"  {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    raise
    return all_embeddings


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_all_papers(data_dir: Path) -> list[dict]:
    """从 metadata 目录加载所有论文."""
    meta_dir = data_dir / "metadata"
    if not meta_dir.exists():
        print(f"错误: 元数据目录不存在: {meta_dir}")
        print("请先运行: python scripts/crawl_papers.py --no-pdf")
        sys.exit(1)

    papers = []
    for json_file in sorted(meta_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            papers.extend(json.load(f))
    return papers


def paper_text(paper: dict) -> str:
    """拼接论文文本用于 embedding."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    return f"{title}\n{abstract}" if abstract else title


def paper_metadata(paper: dict) -> dict:
    """提取存入 ChromaDB 的元数据 (值必须为 str/int/float)."""
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        authors = "; ".join(authors)
    keywords = paper.get("keywords", [])
    if isinstance(keywords, list):
        keywords = "; ".join(keywords)
    return {
        "title": paper.get("title", ""),
        "authors": authors,
        "venue": paper.get("venue", ""),
        "year": int(paper.get("year", 0)),
        "pdf_url": paper.get("pdf_url", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "abstract": (paper.get("abstract", "") or "")[:500],
    }


# ---------------------------------------------------------------------------
# 索引管理
# ---------------------------------------------------------------------------

def get_collection(data_dir: Path) -> chromadb.Collection:
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def cmd_build(args):
    """构建向量索引."""
    data_dir = Path(args.data)
    papers = load_all_papers(data_dir)
    if not papers:
        print("没有找到论文数据")
        return

    # 过滤无文本的论文
    valid = [(p, paper_text(p)) for p in papers if paper_text(p).strip()]
    print(f"加载了 {len(papers)} 篇论文, 其中 {len(valid)} 篇有文本可索引")

    oai = get_openai_client(args.api_key, args.base_url)
    collection = get_collection(data_dir)

    # 检查已有 IDs
    existing_ids = set()
    try:
        result = collection.get()
        existing_ids = set(result["ids"])
    except Exception:
        pass

    to_add = [(p, t) for p, t in valid if p["id"] not in existing_ids]
    if not to_add:
        print("所有论文已在索引中，无需更新")
        return

    print(f"需要索引 {len(to_add)} 篇新论文 (跳过 {len(valid) - len(to_add)} 篇已有)")
    print(f"预计消耗 embedding tokens: ~{sum(len(t.split()) for _, t in to_add) * 1.3:.0f}")

    texts = [t for _, t in to_add]
    ids = [p["id"] for p, _ in to_add]
    metadatas = [paper_metadata(p) for p, _ in to_add]

    # 分批 embedding 并插入 (每批独立，失败不影响已完成的批次)
    batch_size = EMBEDDING_BATCH_SIZE
    success_count = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="批"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        try:
            embeddings = embed_texts(oai, batch_texts, model=args.model)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_meta,
                documents=batch_texts,
            )
            success_count += len(batch_ids)
        except Exception as e:
            print(f"\n  批次 {i//batch_size + 1} 失败: {e}")
            print(f"  已成功索引 {success_count} 篇，可重新运行以继续")
            break

    total = collection.count()
    print(f"索引构建完成! 共 {total} 篇论文")


def cmd_update(args):
    """增量更新索引 (与 build 逻辑相同，会自动跳过已有)."""
    cmd_build(args)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def search_local(collection: chromadb.Collection, oai: OpenAI, query: str,
                 top_k: int, venue: str | None, year: int | None,
                 model: str = DEFAULT_EMBEDDING_MODEL) -> list[dict]:
    """本地向量检索."""
    query_emb = embed_texts(oai, [query], model=model)[0]

    where_filters = {}
    if venue:
        where_filters["venue"] = venue.upper()
    if year:
        where_filters["year"] = year

    kwargs = {
        "query_embeddings": [query_emb],
        "n_results": top_k,
        "include": ["metadatas", "distances", "documents"],
    }
    if where_filters:
        if len(where_filters) == 1:
            key, val = next(iter(where_filters.items()))
            kwargs["where"] = {key: val}
        else:
            kwargs["where"] = {"$and": [{k: v} for k, v in where_filters.items()]}

    results = collection.query(**kwargs)

    papers = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine distance → similarity
        papers.append({
            "id": doc_id,
            "score": round(score, 4),
            "source": "local",
            **meta,
        })
    return papers


def search_live(query: str, top_k: int, venue: str | None, year: int | None) -> list[dict]:
    """Semantic Scholar 实时检索."""
    params = {
        "query": query,
        "limit": min(top_k, 100),
        "fields": S2_FIELDS,
    }
    if venue:
        params["venue"] = venue
    if year:
        params["year"] = str(year)

    try:
        resp = requests.get(S2_SEARCH_API, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(3)
            resp = requests.get(S2_SEARCH_API, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [警告] Semantic Scholar API 调用失败: {e}")
        return []

    data = resp.json()
    papers = []
    for item in data.get("data", []):
        authors = [a.get("name", "") for a in (item.get("authors") or [])]
        pdf_info = item.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url", "") or ""
        ext_ids = item.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv", "")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append({
            "id": item.get("paperId", ""),
            "score": 0.0,
            "source": "live",
            "title": item.get("title", ""),
            "authors": "; ".join(authors),
            "venue": item.get("venue", ""),
            "year": item.get("year", 0) or 0,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
            "abstract": (item.get("abstract", "") or "")[:500],
        })
    return papers


def display_results(papers: list[dict], query: str):
    """格式化显示检索结果."""
    local_count = sum(1 for p in papers if p["source"] == "local")
    live_count = sum(1 for p in papers if p["source"] == "live")

    print(f'\nQuery: "{query}"')
    src_parts = []
    if local_count:
        src_parts.append(f"本地: {local_count}")
    if live_count:
        src_parts.append(f"实时: {live_count}")
    print(f"找到 {len(papers)} 条结果 ({', '.join(src_parts)})\n")

    # 表头
    print(f"{'#':>3}  {'Score':>5}  {'Source':6}  {'Venue':8}  {'Year':4}  Title")
    print("-" * 90)

    for i, p in enumerate(papers, 1):
        score_str = f"{p['score']:.2f}" if p["score"] > 0 else "  -  "
        title = p["title"][:60] + ("..." if len(p["title"]) > 60 else "")
        venue = p.get("venue", "")[:8]
        year = p.get("year", "")
        source = p["source"]
        print(f"{i:>3}  {score_str:>5}  {source:6}  {venue:8}  {year!s:4}  {title}")

    return papers


def interactive_detail(papers: list[dict], data_dir: Path):
    """交互式查看论文详情."""
    while True:
        print("\n输入序号查看详情, 'd 序号' 下载 PDF, 'q' 退出:")
        try:
            inp = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not inp or inp.lower() == "q":
            break

        download = False
        if inp.lower().startswith("d "):
            download = True
            inp = inp[2:].strip()

        try:
            idx = int(inp) - 1
            if idx < 0 or idx >= len(papers):
                print("序号超出范围")
                continue
        except ValueError:
            print("请输入数字序号")
            continue

        p = papers[idx]
        print(f"\n{'='*80}")
        print(f"标题: {p['title']}")
        print(f"作者: {p['authors']}")
        print(f"会议: {p.get('venue', '')} {p.get('year', '')}")
        if p.get("arxiv_id"):
            print(f"ArXiv: https://arxiv.org/abs/{p['arxiv_id']}")
        if p.get("pdf_url"):
            print(f"PDF:   {p['pdf_url']}")
        print(f"\n摘要:\n{p.get('abstract', '(无)')}")
        print(f"{'='*80}")

        if download and p.get("pdf_url"):
            pdf_dir = data_dir / "pdfs" / "downloaded"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r'[<>:"/\\|?*]', "_", p["title"][:80]).strip()
            filepath = pdf_dir / f"{safe}.pdf"
            if filepath.exists():
                print(f"PDF 已存在: {filepath}")
            else:
                print(f"正在下载 PDF...")
                try:
                    resp = requests.get(p["pdf_url"], timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"已下载: {filepath}")
                except Exception as e:
                    print(f"下载失败: {e}")


def cmd_search(args):
    """执行检索."""
    data_dir = Path(args.data)
    query = " ".join(args.query)
    top_k = args.top
    venue = args.venue
    year = args.year

    all_results = []
    seen_ids = set()

    # 本地检索
    collection = get_collection(data_dir)
    if collection.count() > 0:
        oai = get_openai_client(args.api_key, args.base_url)
        local_results = search_local(collection, oai, query, top_k, venue, year, model=args.model)
        for p in local_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])
    else:
        print("[提示] 本地索引为空，请先运行: python paper_search.py build")

    # 实时检索
    if args.live:
        print("正在从 Semantic Scholar 实时检索...")
        live_results = search_live(query, top_k, venue, year)
        for p in live_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    if not all_results:
        print("未找到相关论文")
        return

    papers = display_results(all_results[:top_k], query)
    interactive_detail(papers, data_dir)


# ---------------------------------------------------------------------------
# RAG: 检索 + LLM 综合分析
# ---------------------------------------------------------------------------

def build_context_from_papers(papers: list[dict]) -> str:
    """将论文列表拼接为 LLM 上下文."""
    parts = []
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract", "") or "(无摘要)"
        parts.append(
            f"[{i}] {p['title']}\n"
            f"    作者: {p.get('authors', '')}\n"
            f"    会议: {p.get('venue', '')} {p.get('year', '')}\n"
            f"    摘要: {abstract}"
        )
    return "\n\n".join(parts)


def ask_llm(oai: OpenAI, question: str, context: str, chat_model: str) -> str:
    """用 LLM 基于论文上下文回答问题."""
    system_prompt = """你是一个学术研究助手。用户会给你一个研究问题，以及一组从学术会议论文库中检索到的相关论文（包含标题、作者、会议、摘要）。

请基于这些论文的信息回答用户的问题。要求：
1. 综合多篇论文的信息给出回答，不要只看一篇
2. 回答中引用具体论文时，使用 [编号] 格式（如 [1], [3]）
3. 如果论文信息不足以完全回答问题，明确指出哪些方面缺少证据
4. 区分"论文中明确提到的"和"你的推断"
5. 如果涉及 SOTA 方法或数据对比，尽量给出具体数字
6. 用中文回答"""

    user_msg = f"""## 研究问题
{question}

## 检索到的相关论文
{context}

请基于以上论文回答我的研究问题。"""

    resp = oai.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def cmd_ask(args):
    """检索相关论文 + LLM 综合分析回答问题."""
    data_dir = Path(args.data)
    question = " ".join(args.question)
    top_k = args.top
    venue = args.venue
    year = args.year
    chat_model = args.chat_model

    print(f'问题: "{question}"')
    print(f"正在检索相关论文...")

    all_results = []
    seen_ids = set()

    # 本地检索
    collection = get_collection(data_dir)
    if collection.count() > 0:
        oai = get_openai_client(args.api_key, args.base_url)
        local_results = search_local(collection, oai, question, top_k, venue, year, model=args.model)
        for p in local_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])
    else:
        print("[提示] 本地索引为空，请先运行: python paper_search.py build")
        oai = get_openai_client(args.api_key, args.base_url)

    # 实时检索
    if args.live:
        print("同时从 Semantic Scholar 实时检索...")
        live_results = search_live(question, top_k, venue, year)
        for p in live_results:
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])

    if not all_results:
        print("未找到相关论文，无法回答")
        return

    papers = all_results[:top_k]
    print(f"找到 {len(papers)} 篇相关论文，正在分析...\n")

    # 构建上下文并调用 LLM
    context = build_context_from_papers(papers)
    answer = ask_llm(oai, question, context, chat_model)

    # 输出回答
    print("=" * 80)
    print("回答:")
    print("=" * 80)
    print(answer)
    print("=" * 80)

    # 显示引用的论文列表
    print(f"\n参考论文池 ({len(papers)} 篇):")
    for i, p in enumerate(papers, 1):
        venue_str = p.get("venue", "")
        year_str = p.get("year", "")
        print(f"  [{i}] {p['title'][:70]}  ({venue_str} {year_str})")

    # 进入交互模式，可以继续追问
    print("\n可以继续追问（基于同一批论文），输入 'q' 退出:")
    while True:
        try:
            follow_up = input("\n追问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not follow_up or follow_up.lower() == "q":
            break

        print("正在分析...\n")
        answer = ask_llm(oai, follow_up, context, chat_model)
        print("=" * 80)
        print(answer)
        print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="论文语义检索工具")
    parser.add_argument("--data", "-d", default="papers", help="数据目录 (默认: papers/)")
    parser.add_argument("--api-key", help="OpenAI API key (也可通过 OPENAI_API_KEY 环境变量设置)")
    parser.add_argument("--base-url", help="OpenAI API base URL (也可通过 OPENAI_BASE_URL 环境变量设置)")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help=f"Embedding 模型名 (默认: {DEFAULT_EMBEDDING_MODEL})")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_parser = sub.add_parser("build", help="构建/重建向量索引")

    # update
    update_parser = sub.add_parser("update", help="增量更新索引")

    # search
    search_parser = sub.add_parser("search", help="检索论文")
    search_parser.add_argument("query", nargs="+", help="检索关键词/问题")
    search_parser.add_argument("--top", type=int, default=20, help="返回结果数 (默认: 20)")
    search_parser.add_argument("--venue", help="按会议过滤 (如 NeurIPS, ICLR)")
    search_parser.add_argument("--year", type=int, help="按年份过滤")
    search_parser.add_argument("--live", action="store_true", help="同时从 Semantic Scholar 实时检索")

    # ask
    ask_parser = sub.add_parser("ask", help="基于论文池回答研究问题 (RAG)")
    ask_parser.add_argument("question", nargs="+", help="研究问题")
    ask_parser.add_argument("--top", type=int, default=30, help="检索论文数 (默认: 30)")
    ask_parser.add_argument("--venue", help="按会议过滤")
    ask_parser.add_argument("--year", type=int, help="按年份过滤")
    ask_parser.add_argument("--live", action="store_true", help="同时从 Semantic Scholar 实时检索")
    ask_parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help=f"Chat 模型名 (默认: {DEFAULT_CHAT_MODEL})")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "update":
        cmd_update(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
