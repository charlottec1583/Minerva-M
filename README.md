# Minerva - Academic Paper Search & RAG System

A hybrid paper retrieval system covering **NeurIPS, ICML, ICLR, ACL, EMNLP** (2024-2025).  
Combines local vector search (ChromaDB) with Semantic Scholar live search, plus LLM-powered Q&A.

## Features

- **Paper Crawling** — Fetch metadata from Semantic Scholar (ICLR/NeurIPS/ICML) and ACL Anthology (ACL/EMNLP), with optional PDF download
- **Vector Search** — Build a local ChromaDB index using OpenAI-compatible embedding APIs, supporting venue/year filtering
- **Live Search** — Real-time Semantic Scholar search as a supplement to local index
- **RAG Q&A** — Ask research questions; the system retrieves relevant papers and generates answers via LLM
- **Web UI** — Gradio-based interface with search and Q&A tabs

## Quick Start

### 1. Install Dependencies

```bash
pip install chromadb openai requests tqdm gradio
```

### 2. Crawl Papers (Optional — metadata is included in this repo)

```bash
# Crawl all supported venues (2024-2025), metadata only
python scripts/crawl_papers.py --no-pdf

# Crawl specific venues/years with PDF download
python scripts/crawl_papers.py --venues ICLR NeurIPS --years 2025 -o papers
```

### 3. Build Vector Index

```bash
python scripts/paper_search.py build \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL" \
    -d papers
```

The index is saved locally in `papers/index/` and reused across sessions. Incremental — re-running only indexes new papers.

### 4. Search Papers (CLI)

```bash
# Local vector search
python scripts/paper_search.py search \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL" \
    -q "jailbreak attacks on large language models" \
    --top 10

# With venue/year filter
python scripts/paper_search.py search \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL" \
    -q "multi-turn adversarial attacks" \
    --venue ICLR --year 2025
```

### 5. Ask Research Questions (CLI)

```bash
python scripts/paper_search.py ask \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL" \
    -q "What are the SOTA methods for multi-turn jailbreak attacks? What ASR do they achieve?" \
    --chat-model "YOUR_CHAT_MODEL"
```

### 6. Web UI (Gradio)

```bash
python scripts/paper_search_ui.py \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL"
```

Opens `http://localhost:7860` in your browser with two tabs:
- **Paper Search** — keyword search with filters, results table, and paper details
- **Research Q&A** — ask questions, get LLM answers with cited references, follow-up support

> **Note:** If you're behind a proxy, set `NO_PROXY=localhost,127.0.0.1` before launching to avoid connection errors.

## For Collaborators

This repo includes pre-crawled **metadata** (`papers/metadata/` and `papers/summary.csv`) so you don't need to re-crawl.

To get started, just build the vector index:

```bash
python scripts/paper_search.py build \
    --api-key "YOUR_API_KEY" \
    --base-url "YOUR_API_BASE_URL" \
    --model "YOUR_EMBEDDING_MODEL" \
    -d papers
```

This takes a few minutes and requires an embedding API. The index is stored locally in `papers/index/` (excluded from git due to size).

## Project Structure

```
Minerva/
├── scripts/
│   ├── crawl_papers.py        # Paper metadata & PDF crawler
│   ├── paper_search.py        # Core search engine (CLI)
│   └── paper_search_ui.py     # Gradio web interface
├── papers/
│   ├── metadata/              # Per-venue JSON metadata (tracked)
│   ├── summary.csv            # All papers summary (tracked)
│   ├── index/                 # ChromaDB vector index (local only)
│   └── pdfs/                  # Downloaded PDFs (local only)
├── .gitignore
└── README.md
```

## Supported Venues & Data Sources

| Venue | Source | Years |
|-------|--------|-------|
| ICLR | Semantic Scholar Bulk API | 2024, 2025 |
| NeurIPS | Semantic Scholar Bulk API | 2024, 2025 |
| ICML | Semantic Scholar Bulk API | 2024, 2025 |
| ACL | ACL Anthology (GitHub XML) | 2024, 2025 |
| EMNLP | ACL Anthology (GitHub XML) | 2024, 2025 |

## API Requirements

This system requires access to an **OpenAI-compatible API** for:
- **Embedding** — generating vector representations of paper abstracts
- **Chat** (optional) — powering the RAG Q&A feature

Pass your credentials via `--api-key` and `--base-url` command-line arguments. API keys are **never** stored in code or config files.
