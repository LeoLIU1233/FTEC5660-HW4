"""Reproduce the Task 5 public-test ablation study.

Run from the repository root:

    python task5_ablation.py

The script evaluates retrieval only, so it does not need a DeepSeek key.  It
starts with the naive B0 required by the handout and changes one component per
row.  V7 imports the submitted retrieve() function and therefore reproduces the
configuration used to create results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import hw4
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "public_test"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def plain_text(html: str, preserve_tables: bool) -> str:
    """B0's BeautifulSoup text, optionally swapping tables for table_to_text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()
    if preserve_tables:
        for table in list(soup.find_all("table")):
            replacement = soup.new_string(f"\n{hw4.table_to_text(str(table))}\n")
            table.replace_with(replacement)
    return soup.get_text(" ", strip=True)


def recursive_chunks(
    corpus: dict[str, str],
    docs_meta: dict[str, dict],
    preserve_tables: bool,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=hw4.MAX_CHUNK_CHARS,
        chunk_overlap=0,
        length_function=len,
    )
    chunks: list[Document] = []
    for source, html in sorted(corpus.items()):
        meta = docs_meta[source]
        metadata = {
            "source": source,
            "company": meta["company"],
            "fiscal_year": meta["fiscal_year"],
            "doc_type": meta["doc_type"],
        }
        text = plain_text(html, preserve_tables=preserve_tables)
        for content in splitter.split_text(text):
            if content.strip():
                chunks.append(Document(page_content=content.strip(), metadata=dict(metadata)))
    return chunks


def naive_index(chunks: list[Document]) -> dict:
    matrix = BM25Okapi([
        re.findall(r"\S+", chunk.page_content.casefold())
        for chunk in chunks
    ])
    by_company: dict[str, list[int]] = defaultdict(list)
    for pos, chunk in enumerate(chunks):
        by_company[hw4._name_key(chunk.metadata.get("company"))].append(pos)
    return {"chunks": chunks, "matrix": matrix, "by_company": dict(by_company)}


def top_positions(scores: list[float], positions: list[int]) -> list[int]:
    return sorted(positions, key=lambda pos: float(scores[pos]), reverse=True)[: hw4.MAX_RETRIEVED]


def parsed_positions(index: dict, parsed: dict) -> list[int]:
    """Metadata filter for V3-V6, with a whole-corpus fallback to ensure 'never []'."""
    candidates = hw4._candidate_positions(index, parsed)
    return candidates or list(range(len(index["chunks"])))


def retrieve_v3(index: dict, question: str) -> list[Document]:
    parsed = hw4.parse_query(question)
    positions = parsed_positions(index, parsed)
    query_tokens = re.findall(r"\S+", question.casefold())
    scores = index["matrix"].get_scores(query_tokens)
    return [index["chunks"][pos] for pos in top_positions(scores, positions)]


def retrieve_v4(index: dict, question: str) -> list[Document]:
    parsed = hw4.parse_query(question)
    positions = parsed_positions(index, parsed)
    scores = index["matrix"].get_scores(hw4._expanded_tokens(question))
    return [index["chunks"][pos] for pos in top_positions(scores, positions)]


def boosted_scores(index: dict, question: str, parsed: dict) -> list[float]:
    scores = [float(value) for value in index["matrix"].get_scores(hw4._expanded_tokens(question))]
    phrases = hw4._concept_phrases(question)
    year = parsed.get("fiscal_year")
    for pos, chunk in enumerate(index["chunks"]):
        raw = chunk.page_content.casefold()
        for phrase in phrases:
            if phrase.casefold() in raw:
                scores[pos] += 7.0 + min(len(phrase), 30) / 10
        if isinstance(year, int):
            if re.search(rf"\b{year}\b", raw):
                scores[pos] += 3.0
            if chunk.metadata.get("fiscal_year") == year:
                scores[pos] += 0.75
    return scores


def retrieve_v5(index: dict, question: str) -> list[Document]:
    parsed = hw4.parse_query(question)
    positions = parsed_positions(index, parsed)
    scores = boosted_scores(index, question, parsed)
    return [index["chunks"][pos] for pos in top_positions(scores, positions)]


def retrieve_v6(index: dict, question: str) -> list[Document]:
    """V5 plus a per-company reservation for comparison questions."""
    parsed = hw4.parse_query(question)
    companies = parsed.get("companies") or []
    positions = parsed_positions(index, parsed)
    scores = boosted_scores(index, question, parsed)
    if not companies:
        return [index["chunks"][pos] for pos in top_positions(scores, positions)]

    by_company: dict[str, list[int]] = defaultdict(list)
    for pos in positions:
        company = hw4._name_key(index["chunks"][pos].metadata.get("company"))
        by_company[company].append(pos)
    company_keys = [hw4._name_key(company) for company in companies]
    if any(key not in by_company for key in company_keys):
        return [index["chunks"][pos] for pos in top_positions(scores, positions)]

    chosen: list[int] = []
    seen: set[int] = set()
    quota = 1 if len(company_keys) >= hw4.MAX_RETRIEVED else min(
        2, hw4.MAX_RETRIEVED // max(len(company_keys), 1)
    )
    for key in company_keys:
        for pos in top_positions(scores, by_company[key])[:quota]:
            if pos not in seen and len(chosen) < hw4.MAX_RETRIEVED:
                chosen.append(pos)
                seen.add(pos)
    for pos in top_positions(scores, positions):
        if pos not in seen and len(chosen) < hw4.MAX_RETRIEVED:
            chosen.append(pos)
            seen.add(pos)
    return [index["chunks"][pos] for pos in chosen]


def retrieval_score(chunks: list[Document], gold: dict) -> bool | str:
    return hw4.retrieval_hit(chunks, gold) if gold["evidence"] else ""


def parse_field_accuracy(parsed: Any, gold: dict) -> float:
    return hw4.parse_score(parsed, gold)


def evaluate(
    name: str,
    change: str,
    chunks: list[Document],
    index: dict,
    questions: dict[str, str],
    ground_truth: dict[str, dict],
    retriever: Callable[[dict, str], list[Document]] | None,
    parse_queries: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for qid, question in questions.items():
        gold = ground_truth[qid]
        parsed = hw4.parse_query(question) if parse_queries else None
        retrieved = retriever(index, question) if retriever else []
        rows.append({
            "id": qid,
            "category": gold["category"],
            "retrieved": ";".join(
                str(chunk.metadata.get("source", "")) for chunk in retrieved
            ),
            "hit": retrieval_score(retrieved, gold) if retrieved else "",
            "parse_accuracy": parse_field_accuracy(parsed, gold) if parse_queries else "",
            "nonempty": bool(retrieved),
        })

    answerable = [row for row in rows if ground_truth[row["id"]]["evidence"]]
    unanswerable = [row for row in rows if not ground_truth[row["id"]]["evidence"]]
    kept = sum(row["nonempty"] for row in answerable) / len(answerable)
    rejected = sum(not row["nonempty"] for row in unanswerable) / len(unanswerable)
    recall = sum(row["hit"] is True for row in answerable) / len(answerable)
    parse_acc = (
        sum(row["parse_accuracy"] for row in rows) / len(rows)
        if parse_queries else None
    )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if ground_truth[row["id"]]["evidence"]:
            by_category[row["category"]].append(row)

    summary = {
        "variant": name,
        "change": change,
        "chunks": len(chunks),
        "parse_accuracy": "" if parse_acc is None else round(parse_acc, 6),
        "recall_at_5": round(recall, 6),
        "answerable_kept": round(kept, 6),
        "unanswerable_rejected": round(rejected, 6),
        "abstention_score": round(max(0.0, kept + rejected - 1.0), 6),
        "fact_recall": round(
            sum(row["hit"] is True for row in by_category["fact"]) / len(by_category["fact"]), 6
        ),
        "table_recall": round(
            sum(row["hit"] is True for row in by_category["table"]) / len(by_category["table"]), 6
        ),
        "comparison_recall": round(
            sum(row["hit"] is True for row in by_category["comparison"])
            / len(by_category["comparison"]), 6
        ),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "task5_ablation.csv")
    parser.add_argument(
        "--details",
        type=Path,
        default=ROOT / "task5_ablation_details.csv",
    )
    args = parser.parse_args()

    corpus = hw4.load_corpus(args.data_folder / "corpus")
    docs_meta = load_json(args.data_folder / "docs.json")
    questions = load_json(args.data_folder / "questions.json")
    ground_truth = load_json(args.data_folder / "ground_truth.json")

    # Register company names before B0's parse-free retrieval; later parse_query()
    # calls use this corpus-derived vocabulary rather than hard-coded company names.
    final_chunks = [
        chunk
        for source, html in sorted(corpus.items())
        for chunk in hw4.chunk_document(source, html)
    ]
    b0_chunks = recursive_chunks(corpus, docs_meta, preserve_tables=False)
    v1_chunks = recursive_chunks(corpus, docs_meta, preserve_tables=True)
    b0_index = naive_index(b0_chunks)
    v1_index = naive_index(v1_chunks)
    v2_index = naive_index(final_chunks)
    final_index = hw4.build_index(final_chunks)

    def whole_corpus(index: dict, question: str) -> list[Document]:
        query_tokens = re.findall(r"\S+", question.casefold())
        scores = index["matrix"].get_scores(query_tokens)
        return [index["chunks"][pos] for pos in top_positions(scores, list(range(len(index["chunks"]))))]

    variants = [
        (
            "B0", "Naive RAG baseline: raw BeautifulSoup text, recursive split, "
            "lower-cased whitespace BM25, whole-corpus top-5, no parsing, never []",
            b0_chunks, b0_index, whole_corpus, False,
        ),
        (
            "V1", "+ Preserve table rows and column labels with table_to_text()",
            v1_chunks, v1_index, whole_corpus, False,
        ),
        (
            "V2", "+ Section/table-aware chunks and contextual chunk headers",
            final_chunks, v2_index, whole_corpus, False,
        ),
        (
            "V3", "+ parse_query() and metadata filtering by company/year/doc type",
            final_chunks, v2_index, retrieve_v3, True,
        ),
        (
            "V4", "+ normalized tokenization and synonym expansion",
            final_chunks, final_index, retrieve_v4, True,
        ),
        (
            "V5", "+ concept-phrase and named-year score boosts",
            final_chunks, final_index, retrieve_v5, True,
        ),
        (
            "V6", "+ per-company result quota for comparison questions",
            final_chunks, final_index, retrieve_v6, True,
        ),
        (
            "V7", "+ abstention guards (unknown company/year or missing metric), "
            "matching submitted retrieve()",
            final_chunks, final_index, hw4.retrieve, True,
        ),
    ]

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for name, change, chunks, index, retriever, parse_queries in variants:
        summary, rows = evaluate(
            name, change, chunks, index, questions, ground_truth, retriever, parse_queries
        )
        summaries.append(summary)
        for row in rows:
            details.append({"variant": name, **row})
        print(
            f"{name}: recall@5={summary['recall_at_5']:.1%}, "
            f"kept={summary['answerable_kept']:.1%}, "
            f"rejected={summary['unanswerable_rejected']:.1%}, "
            f"chunks={summary['chunks']}"
        )

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with args.details.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.details}")


if __name__ == "__main__":
    main()
