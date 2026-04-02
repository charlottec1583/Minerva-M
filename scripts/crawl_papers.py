#!/usr/bin/env python3
"""
学术会议论文爬取脚本
支持会议: NeurIPS, ICML, ICLR (via Semantic Scholar), ACL, EMNLP (via ACL Anthology)
功能: 爬取元数据 + 下载 PDF, 支持增量更新
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# Semantic Scholar 会议名称 (用于 venue 参数)
SEMANTIC_SCHOLAR_VENUES = {
    "ICLR":    "ICLR",
    "NeurIPS": "NeurIPS",
    "ICML":    "ICML",
}

S2_BULK_API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,authors,abstract,venue,year,externalIds,openAccessPdf"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"

# ACL Anthology: XML 文件名 → 要抓取的 volume ID 列表
# XML 文件格式: {year}.{collection}.xml, volume 在 XML 内部
# collection_id 和 volume_ids 的含义:
#   (collection="acl", volumes=["long","short"]) → 2024.acl.xml 中的 long/short 卷
#   (collection="findings", volumes=["acl"])     → 2024.findings.xml 中的 acl 卷
ACL_ANTHOLOGY_SOURCES = {
    "ACL": [
        # (xml_collection, [volume_ids_to_include])
        ("acl", ["long", "short"]),
        ("findings", ["acl"]),
    ],
    "EMNLP": [
        ("emnlp", ["main"]),
        ("findings", ["emnlp"]),
    ],
}

ACL_ANTHOLOGY_XML_BASE = (
    "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml"
)
ACL_ANTHOLOGY_PDF_BASE = "https://aclanthology.org"

DEFAULT_VENUES = list(SEMANTIC_SCHOLAR_VENUES.keys()) + list(ACL_ANTHOLOGY_SOURCES.keys())
DEFAULT_YEARS = [2024, 2025]

REQUEST_DELAY = 1.0  # seconds between PDF downloads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic Scholar 爬取器 (ICLR, NeurIPS, ICML)
# ---------------------------------------------------------------------------

def fetch_semantic_scholar_papers(venue: str, year: int) -> list[dict]:
    """通过 Semantic Scholar Bulk API 获取会议论文元数据."""
    s2_venue = SEMANTIC_SCHOLAR_VENUES[venue]
    logger.info(f"正在从 Semantic Scholar 获取 {venue} {year} 的论文...")

    papers = []
    token = None  # pagination token

    while True:
        params = {
            "query": "",
            "venue": s2_venue,
            "year": str(year),
            "fields": S2_FIELDS,
            "limit": 1000,
        }
        if token:
            params["token"] = token

        try:
            resp = requests.get(S2_BULK_API, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("  S2 API 限速，等待 5 秒...")
                time.sleep(5)
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"  Semantic Scholar API 调用失败: {e}")
            break

        data = resp.json()
        batch = data.get("data", [])
        if not batch:
            break

        for item in batch:
            # 提取 PDF URL: 优先 openAccessPdf, 其次 ArXiv
            pdf_info = item.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url", "") or ""
            ext_ids = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv", "")

            if not pdf_url and arxiv_id:
                pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"

            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]

            paper_id = item.get("paperId", "")
            papers.append({
                "id": paper_id,
                "title": item.get("title", ""),
                "authors": authors,
                "abstract": item.get("abstract", "") or "",
                "keywords": [],
                "venue": venue,
                "year": year,
                "pdf_url": pdf_url,
                "arxiv_id": arxiv_id,
                "local_pdf_path": "",
            })

        token = data.get("token")
        if not token:
            break
        time.sleep(1)  # 限速

    logger.info(f"  -> 获取到 {len(papers)} 篇论文")
    return papers


# ---------------------------------------------------------------------------
# ACL Anthology 爬取器 (via GitHub XML)
# ---------------------------------------------------------------------------

def fetch_acl_anthology_papers(venue: str, year: int) -> list[dict]:
    """通过 ACL Anthology GitHub XML 获取论文元数据."""
    sources = ACL_ANTHOLOGY_SOURCES[venue]
    all_papers = []

    for collection, target_volumes in sources:
        xml_file = f"{year}.{collection}.xml"
        xml_url = f"{ACL_ANTHOLOGY_XML_BASE}/{xml_file}"
        logger.info(f"正在从 ACL Anthology 获取 {xml_file} (volumes: {target_volumes})...")

        try:
            resp = requests.get(xml_url, timeout=30)
            if resp.status_code == 404:
                logger.warning(f"  -> {xml_file} 尚未发布 (404)")
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"  -> 下载 XML 失败: {e}")
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            logger.error(f"  -> 解析 XML 失败: {e}")
            continue

        collection_id = root.get("id", f"{year}.{collection}")

        for volume_el in root.findall("volume"):
            vol_id = volume_el.get("id", "")
            if vol_id not in target_volumes:
                continue

            for paper_el in volume_el.findall("paper"):
                paper_id_num = paper_el.get("id", "")
                if paper_id_num == "0":
                    continue  # skip front-matter

                # 构建 anthology ID: e.g. "2024.acl-long.1"
                anthology_id = f"{collection_id}-{vol_id}.{paper_id_num}"

                title_el = paper_el.find("title")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""

                authors = []
                for author_el in paper_el.findall("author"):
                    first = author_el.findtext("first", "").strip()
                    last = author_el.findtext("last", "").strip()
                    authors.append(f"{first} {last}".strip())

                abstract_el = paper_el.find("abstract")
                abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else ""

                url_el = paper_el.find("url")
                url_text = url_el.text.strip() if url_el is not None and url_el.text else anthology_id
                pdf_url = f"{ACL_ANTHOLOGY_PDF_BASE}/{url_text}.pdf"

                all_papers.append({
                    "id": anthology_id,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "keywords": [],
                    "venue": venue,
                    "year": year,
                    "pdf_url": pdf_url,
                    "local_pdf_path": "",
                })

    logger.info(f"  -> {venue} {year} 共获取 {len(all_papers)} 篇论文")
    return all_papers


# ---------------------------------------------------------------------------
# PDF 下载器
# ---------------------------------------------------------------------------

def download_pdfs(papers: list[dict], pdf_dir: Path, error_log: Path) -> None:
    """下载论文 PDF，跳过已存在的文件."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    to_download = []
    for p in papers:
        if not p["pdf_url"]:
            continue
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", p["title"][:80]).strip()
        filename = f"{p['id']}_{safe_name}.pdf"
        filepath = pdf_dir / filename
        p["local_pdf_path"] = str(filepath)
        if filepath.exists():
            continue
        to_download.append((p, filepath))

    if not to_download:
        logger.info(f"  PDF 目录 {pdf_dir}: 无需下载新文件")
        return

    logger.info(f"  开始下载 {len(to_download)} 个 PDF 到 {pdf_dir}")
    for paper, filepath in tqdm(to_download, desc="下载PDF", unit="篇"):
        try:
            resp = requests.get(paper["pdf_url"], timeout=60, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            errors.append({"id": paper["id"], "title": paper["title"], "url": paper["pdf_url"], "error": str(e)})
            logger.warning(f"  下载失败: {paper['title'][:50]}... - {e}")

    if errors:
        with open(error_log, "a", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
        logger.warning(f"  {len(errors)} 个 PDF 下载失败，详见 {error_log}")


# ---------------------------------------------------------------------------
# 元数据管理
# ---------------------------------------------------------------------------

def load_existing_metadata(meta_path: Path) -> dict[str, dict]:
    """加载已有的元数据 JSON，返回 {id: paper} 字典."""
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: p for p in data}


def save_metadata(papers: list[dict], meta_path: Path) -> None:
    """保存元数据到 JSON 文件."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    logger.info(f"  元数据已保存: {meta_path} ({len(papers)} 篇)")


def save_summary_csv(all_papers: list[dict], csv_path: Path) -> None:
    """保存所有论文的汇总 CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "title", "authors", "abstract", "keywords", "venue", "year", "pdf_url", "arxiv_id", "local_pdf_path"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in all_papers:
            row = dict(p)
            row["authors"] = "; ".join(row["authors"]) if isinstance(row["authors"], list) else row["authors"]
            row["keywords"] = "; ".join(row["keywords"]) if isinstance(row["keywords"], list) else row["keywords"]
            row.setdefault("arxiv_id", "")
            writer.writerow(row)
    logger.info(f"汇总 CSV 已保存: {csv_path} ({len(all_papers)} 篇)")


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def crawl_venue_year(venue: str, year: int, output_dir: Path, download_pdf: bool) -> list[dict]:
    """爬取单个会议单个年份的论文."""
    meta_dir = output_dir / "metadata"
    meta_path = meta_dir / f"{venue.lower()}_{year}.json"
    pdf_dir = output_dir / "pdfs" / f"{venue.lower()}_{year}"
    error_log = output_dir / "errors.log"

    # 获取新论文
    if venue in SEMANTIC_SCHOLAR_VENUES:
        new_papers = fetch_semantic_scholar_papers(venue, year)
    elif venue in ACL_ANTHOLOGY_SOURCES:
        new_papers = fetch_acl_anthology_papers(venue, year)
    else:
        logger.error(f"不支持的会议: {venue}")
        return []

    if not new_papers:
        return []

    # 增量合并
    existing = load_existing_metadata(meta_path)
    merged = dict(existing)
    added = 0
    for p in new_papers:
        if p["id"] not in merged:
            merged[p["id"]] = p
            added += 1
        else:
            # 更新已有条目但保留 local_pdf_path
            old_path = merged[p["id"]].get("local_pdf_path", "")
            merged[p["id"]] = p
            if old_path:
                merged[p["id"]]["local_pdf_path"] = old_path

    papers_list = list(merged.values())
    logger.info(f"  {venue} {year}: {added} 篇新论文, 共 {len(papers_list)} 篇")

    # 下载 PDF
    if download_pdf:
        download_pdfs(papers_list, pdf_dir, error_log)

    # 保存元数据
    save_metadata(papers_list, meta_path)
    return papers_list


def main():
    parser = argparse.ArgumentParser(
        description="爬取 NeurIPS/ICML/ICLR/ACL/EMNLP 会议论文"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="papers",
        help="输出目录 (默认: papers/)",
    )
    parser.add_argument(
        "--venues", nargs="+", default=DEFAULT_VENUES,
        choices=DEFAULT_VENUES,
        help=f"要爬取的会议 (默认: {DEFAULT_VENUES})",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help=f"要爬取的年份 (默认: {DEFAULT_YEARS})",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="只下载元数据，不下载 PDF",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_papers = []
    for venue in args.venues:
        for year in args.years:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始爬取: {venue} {year}")
            logger.info(f"{'='*60}")
            papers = crawl_venue_year(venue, year, output_dir, not args.no_pdf)
            all_papers.extend(papers)

    # 保存汇总 CSV
    if all_papers:
        save_summary_csv(all_papers, output_dir / "summary.csv")

    logger.info(f"\n完成! 共爬取 {len(all_papers)} 篇论文, 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
