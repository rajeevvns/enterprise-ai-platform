"""
shared/evaluation/judge.py
--------------------------
LLM-as-judge helpers for RAG evaluation.

Scores a generated answer against an expected answer on a 0–3 scale.
Used by: learning/week06/acmecloud_rag_eval.py (and future eval scripts)
"""

import json
import re
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an impartial evaluator. Score the RAG answer against the expected answer.\n"
    "Return ONLY a JSON object with two keys:\n"
    "  score:  integer 0-3  "
    "(0=wrong or hallucinated, 1=partially correct, 2=mostly correct, 3=fully correct)\n"
    "  reason: one sentence explaining the score\n"
    "Do not add any other text outside the JSON object."
)


# ---------------------------------------------------------------------------
# Per-token cost rates  (override when calling judge_answer if needed)
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_RATES = {
    "gpt-4o":       {"input": 0.003  / 1_000, "output": 0.012 / 1_000},
    "gpt-4o-mini":  {"input": 0.00015 / 1_000, "output": 0.0006 / 1_000},
}


# ---------------------------------------------------------------------------
# judge_answer
# ---------------------------------------------------------------------------

async def judge_answer(
    question: str,
    expected: str,
    actual: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    rates: dict | None = None,
) -> tuple[int, str, dict]:
    """
    Score a generated answer against the expected answer.

    Returns:
        score  (int 0–3)
        reason (str)
        usage  (dict: input_tokens, output_tokens, cost_usd)
    """
    if rates is None:
        rates = DEFAULT_JUDGE_RATES.get(model, DEFAULT_JUDGE_RATES["gpt-4o"])

    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"RAG answer: {actual}"
    )

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )

    raw = (resp.choices[0].message.content or "").strip()

    # Strip optional markdown fences the model may add
    json_str = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(json_str)
        score  = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        score  = 0
        reason = f"[parse error] raw response: {raw[:200]}"

    token_usage = resp.usage
    in_tok  = token_usage.prompt_tokens     if token_usage else 0
    out_tok = token_usage.completion_tokens if token_usage else 0
    usage = {
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      in_tok * rates["input"] + out_tok * rates["output"],
    }

    return score, reason, usage


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def score_summary(results: list[dict]) -> dict:
    """
    Compute aggregate judge metrics from a list of result dicts.

    Each result dict must have: judge_score (int), difficulty (str).

    Returns dict with: avg_score, score_distribution, by_difficulty.
    """
    scores = [r["judge_score"] for r in results]
    avg    = sum(scores) / len(scores) if scores else 0.0

    dist = {str(s): 0 for s in range(4)}
    for s in scores:
        dist[str(s)] += 1

    by_diff: dict[str, list[int]] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["judge_score"])

    return {
        "avg_score":          round(avg, 3),
        "score_distribution": dist,
        "by_difficulty": {
            d: round(sum(ss) / len(ss), 3)
            for d, ss in sorted(by_diff.items())
        },
    }
