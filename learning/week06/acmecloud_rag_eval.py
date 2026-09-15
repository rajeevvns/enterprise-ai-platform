"""
acmecloud_rag_eval.py
---------------------
Week 6 Lab — Naive RAG Foundations & Baseline Evaluation

Applies the naive RAG pipeline to the AcmeCloud corpus and measures
retrieval + generation quality against the 40-question golden set.

Pipeline:
  1. Load & chunk corpus  (shared/data/corpus/acmecloud/)
  2. Embed all chunks     (text-embedding-3-small, single batch)
  3. For each question:   embed query → retrieve top-5 → generate answer
  4. LLM-as-judge:        score each answer 0–3 vs expected
  5. Print KPI table      + save full results to acmecloud_rag_results.json

Run (from any directory):
  export OPENAI_API_KEY=<your-key>
  python learning/week06/acmecloud_rag_eval.py

Optional env var:
  OPENAI_BASE_URL   Override the API base URL (defaults to the Vocareum proxy,
                    https://openai.vocareum.com/v1 — set this to point at the
                    real OpenAI API instead).

Jupyter (avoid asyncio.run in a running loop):
  import acmecloud_rag_eval      # then call:
  results, embed_cost = await acmecloud_rag_eval.run_eval()
"""

import os
import sys
import json
import time
import asyncio
import pathlib
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

# ---------------------------------------------------------------------------
# Path setup — allow imports from shared/ regardless of working directory
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # …/AI-RAG_MP1_PromptLab
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utils.embeddings import (   # noqa: E402
    load_corpus,
    embed_batch,
    embed_corpus,
    retrieve,
    source_recall,
)
from shared.evaluation.judge import judge_answer, score_summary  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR   = REPO_ROOT / "shared/data/corpus/acmecloud"
GOLDEN_FILE  = REPO_ROOT / "shared/data/golden_set/acmecloud/golden_set_40_questions.json"
RESULTS_FILE = pathlib.Path(__file__).parent / "acmecloud_rag_results.json"

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
JUDGE_MODEL     = "gpt-4o"

TOP_K = 5

# Cost rates
EMBED_RATE  = 0.02 / 1_000_000          # text-embedding-3-small: $0.02 / 1M tokens
CHAT_RATES  = {"input": 0.00015 / 1_000, "output": 0.0006 / 1_000}   # gpt-4o-mini

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running"

client = AsyncOpenAI(base_url=OPENAI_BASE_URL)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise assistant for the AcmeCloud Enterprise AI Platform. "
    "Answer the question using ONLY the provided context chunks. "
    "If the answer is not contained in the context, reply with exactly: "
    "\"I cannot answer from the provided context.\""
)


async def generate_answer(question: str, context_chunks: list[dict]) -> tuple[str, dict]:
    """Call gpt-4o-mini with retrieved context. Returns (answer, usage_dict)."""
    context_text = "\n\n---\n\n".join(
        f"[{c['doc_id']}]\n{c['text']}" for c in context_chunks
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]
    t0   = time.perf_counter()
    resp = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
    )
    latency     = time.perf_counter() - t0
    answer      = (resp.choices[0].message.content or "").strip()
    token_usage = resp.usage
    in_tok      = token_usage.prompt_tokens     if token_usage else 0
    out_tok     = token_usage.completion_tokens if token_usage else 0
    usage = {
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      in_tok * CHAT_RATES["input"] + out_tok * CHAT_RATES["output"],
        "latency_s":     round(latency, 3),
    }
    return answer, usage

# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

async def process_one(
    q: dict,
    q_vec: list[float],
    all_chunks: list[dict],
) -> dict:
    retrieved = retrieve(q_vec, all_chunks, top_k=TOP_K)

    answer, gen_usage              = await generate_answer(q["question"], retrieved)
    score, reason, judge_usage     = await judge_answer(
        question=q["question"],
        expected=q["expected_answer"],
        actual=answer,
        client=client,
        model=JUDGE_MODEL,
    )
    recall_info = source_recall(retrieved, q.get("expected_sources", []))

    return {
        "id":               q["id"],
        "difficulty":       q["difficulty"],
        "question_type":    q["question_type"],
        "question":         q["question"],
        "expected_answer":  q["expected_answer"],
        "answerable":       q["answerable"],
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "doc_id":   c["doc_id"],
                "preview":  c["text"][:120],
            }
            for c in retrieved
        ],
        **recall_info,
        "generated_answer": answer,
        "judge_score":      score,
        "judge_reason":     reason,
        "gen_usage":        gen_usage,
        "judge_usage":      judge_usage,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_eval() -> tuple[list[dict], float]:
    """Run the full evaluation. Returns (results, embed_cost_usd)."""

    # 1 — Load & embed corpus
    all_chunks = load_corpus(CORPUS_DIR, strategy="paragraph")

    print(f"Embedding {len(all_chunks)} chunks …")
    t0         = time.perf_counter()
    all_chunks = await embed_corpus(all_chunks, client, EMBED_MODEL)
    embed_time = time.perf_counter() - t0

    # Approximate embed cost (token counts not returned by the embeddings endpoint)
    approx_tokens = sum(len(c["text"].split()) * 1.3 for c in all_chunks)
    embed_cost    = approx_tokens * EMBED_RATE
    print(f"Embedding done in {embed_time:.1f}s  (~${embed_cost:.5f} estimated)\n")

    # 2 — Load golden set
    golden = json.loads(GOLDEN_FILE.read_text())
    print(f"Running {len(golden)} golden questions …\n")

    # 3 — Embed all questions in one batch
    questions     = [q["question"] for q in golden]
    query_vectors = await embed_batch(questions, client, EMBED_MODEL)

    # 4 — Retrieve + generate + judge (all concurrent)
    tasks   = [process_one(q, qv, all_chunks) for q, qv in zip(golden, query_vectors)]
    results = await asyncio.gather(*tasks)

    return list(results), embed_cost

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict], embed_cost: float) -> None:
    total      = len(results)
    recalls    = [r["source_recall"] for r in results]
    avg_recall = sum(recalls) / total
    avg_latency = sum(r["gen_usage"]["latency_s"] for r in results) / total

    gen_cost   = sum(r["gen_usage"]["cost_usd"]   for r in results)
    judge_cost = sum(r["judge_usage"]["cost_usd"] for r in results)
    total_cost = embed_cost + gen_cost + judge_cost

    summary = score_summary(results)

    print("=" * 65)
    print("  ACMECLOUD NAIVE RAG — BASELINE KPI SNAPSHOT")
    print("=" * 65)
    print(f"  Questions evaluated : {total}")
    print(f"  Avg judge score     : {summary['avg_score']:.2f} / 3.00")
    print(f"  Avg source recall   : {avg_recall:.1%}")
    print(f"  Avg gen latency     : {avg_latency:.2f}s")
    print(f"  Total cost          : ${total_cost:.4f}")
    print(f"    ├─ embed          : ${embed_cost:.5f}")
    print(f"    ├─ generation     : ${gen_cost:.4f}")
    print(f"    └─ judge          : ${judge_cost:.4f}")
    print()
    print("  Score by difficulty:")
    for diff, avg in summary["by_difficulty"].items():
        count = sum(1 for r in results if r["difficulty"] == diff)
        print(f"    {diff:<8s} : {avg:.2f} / 3.00  (n={count})")
    print()
    print("  Score distribution:")
    for score, count in sorted(summary["score_distribution"].items()):
        bar = "█" * count
        print(f"    {score}/3 : {bar} {count}")
    print()

    # Per-question table
    print(f"  {'ID':<6} {'Diff':<8} {'Score':>5} {'Recall':>7}  Question")
    print(f"  {'--':<6} {'----':<8} {'-----':>5} {'------':>7}  --------")
    for r in results:
        q_short = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        print(
            f"  {r['id']:<6} {r['difficulty']:<8} {r['judge_score']:>5} "
            f"{r['source_recall']:>6.0%}  {q_short}"
        )
    print()

    # Low-scoring deep-dive
    poor = [r for r in results if r["judge_score"] <= 1]
    if poor:
        print(f"  ⚠  Low-scoring questions (score ≤ 1): {len(poor)}")
        for r in poor:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    Expected : {r['expected_answer'][:120]}")
            print(f"    Got      : {r['generated_answer'][:120]}")
            print(f"    Reason   : {r['judge_reason']}")
            print(f"    Sources needed : {r['expected_sources']}")
            print(f"    Sources hit    : {r['sources_hit']}")

    print()
    print(f"  Full results → {RESULTS_FILE}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    results, embed_cost = await run_eval()
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print_report(results, embed_cost)


if __name__ == "__main__":
    asyncio.run(main())
