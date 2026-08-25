# ADR 0002: Prompting Strategy for Information Extraction

## Status
✅ Accepted

## Context

Our capstone project requires extracting structured data (company, role, years_experience_required) from job postings at scale. Different prompting strategies have different trade-offs in accuracy, cost, and latency. We need to choose a strategy that balances accuracy, performance, and maintainability.

## Decision

We will use a **few-shot prompting strategy with domain-specific examples** for information extraction tasks in the capstone.

## Rationale

### Evidence from MP1 (Prompt Lab Mini-Project)

We conducted a systematic comparison of four prompting strategies on 10 job postings (40 API calls total) using gpt-4o-mini via Vocareum. Results:

| Strategy | Accuracy | Latency | Cost | Parse Success |
|----------|----------|---------|------|---|
| **Few-shot** | **2.9/3 (97%)** | 0.871s | $0.001 | 100% |
| Structured | 2.7/3 (90%) | 1.064s | $0.001 | 100% |
| Chain-of-thought | 2.7/3 (90%) | 1.398s | $0.001 | 100% |
| Zero-shot | 2.6/3 (87%) | 1.028s | $0.000 | 100% |

**Key Finding:** Few-shot achieved the highest accuracy (2.9/3) outperforming all other strategies by 3-10%. Demonstrations proved more powerful than explicit rules or reasoning.

### Why Few-Shot?

1. **Highest Accuracy (97%)** — Achieved 2.9/3 average, the only strategy consistently handling edge cases (null values) correctly

2. **Fast Latency (0.871s)** — Actually the fastest despite longer prompts; acceptable for batch processing

3. **Perfect Edge Case Handling** — The null case (j10) was critical. Few-shot with examples correctly returned null instead of hallucinating years. Without the null example, model would guess "3 years" or "5 years"

4. **Production Ready** — 100% JSON parse rate; no wrapped markdown fences or preamble text issues

5. **Demonstrates > Declares** — The model learns "do this" better from 2-3 worked examples than from explicit rules or system prompts

6. **Generalizable** — Easy to adapt with domain-specific examples for different extraction tasks

### Why NOT Others?

**Structured Strategy**
- More verbose system prompts with explicit JSON schema
- Same or worse accuracy (2.7/3 vs 2.9/3)
- Model doesn't follow explicit rules better than it follows examples
- Higher latency (1.064s)

**Chain-of-Thought**
- Slowest (1.398s, 60% slower than few-shot)
- No accuracy improvement despite reasoning
- Model would reason correctly ("no years stated") then extract a number anyway
- Better suited for multi-step reasoning tasks, not extraction

**Zero-Shot**
- Weakest baseline (2.6/3, 87% accuracy)
- Lacks guidance for edge cases
- No examples to train model behavior

## Consequences

### Positive ✅
- Highest accuracy for extraction tasks (97%)
- Fast enough latency for batch processing
- Minimal cost difference vs other strategies (~$0.001)
- 100% JSON parse success rate
- Easy to extend with domain-specific examples
- Clear, maintainable prompt structure

### Negative ⚠️
- Requires 2-3 high-quality examples
- Must include at least one edge case
- Less "self-documenting" than structured prompts (rules are implicit in examples)
- Requires occasional updates as domain evolves

## Implementation Plan

For the capstone RAG system, we will:

1. **Keep few-shot base format** from MP1 as the foundation
2. **Create domain-specific examples** (3-5 examples matching our job posting schema):
   - Include salary ranges if available
   - Include remote work flags
   - Include location information
   - Always include at least one null case (missing field)
3. **Validate** with a small golden set before production deployment (10-20 test cases)
4. **Monitor** extraction quality and update examples if accuracy dips

### Code Structure

```python
def prompt_few_shot_capstone(job_posting: str) -> list[dict]:
    """Extract job data using few-shot with domain-specific examples."""
    
    examples = """
EXAMPLE 1 (Complete):
Posting: "TechCorp Inc is hiring a Senior Software Engineer with 5+ years Python experience. Salary: $120k-$150k. Remote/Hybrid."
Output: {"company": "TechCorp Inc", "role": "Senior Software Engineer", "years": 5, "salary_min": 120000, "remote": "Hybrid"}

EXAMPLE 2 (Partial):
Posting: "StartupXYZ seeks Data Analyst, 2 years minimum. Remote only."
Output: {"company": "StartupXYZ", "role": "Data Analyst", "years": 2, "salary_min": null, "remote": "Remote"}

EXAMPLE 3 (Null case - CRITICAL):
Posting: "BigBank is hiring Risk Analysts. We welcome all experience levels. Competitive salary."
Output: {"company": "BigBank", "role": "Risk Analyst", "years": null, "salary_min": null, "remote": null}
"""
    
    prompt = f"""Extract structured job data. Follow the examples exactly.

{examples}

Now extract from this posting:
{job_posting}

Return ONLY valid JSON matching the example format."""
    
    return [{'role': 'user', 'content': prompt}]
```

## Alternatives Considered

1. **Structured + JSON Schema** — More explicit but worse results in practice; model doesn't follow rules
2. **Chain-of-Thought** — Better for multi-step reasoning; slower and no accuracy gain for extraction
3. **Fine-Tuning** — Out of scope; requires labeled training data and API costs
4. **Hybrid (Few-Shot + Structured)** — Possible future enhancement combining both approaches
5. **Prompt Optimization** — Can iterate on few-shot examples post-launch if needed

## Related Decisions

- **MP1 Project:** Systematic strategy comparison (Foundations phase, W5)
- **W2 Pattern:** Async batching for 40 parallel API calls
- **W4 Pattern:** Cost tracking per strategy
- **W5 Pattern:** Golden set scoring and LLM-as-judge evaluation

## References

- **MP1 Comparison:** `mp1_comparison.md` — Detailed results table
- **MP1 Writeup:** `mp1_writeup.md` — Reflection on findings
- **MP1 Notebook:** `MP1_Starter_Template.ipynb` — Full implementation
- **MP1 Brief:** `learner/MP1_Brief.md` — Task specification
- **Data:** `data/job_snippets.jsonl`, `data/golden_set.jsonl` — Test cases

## Tags

- `#prompting`
- `#prompt-engineering`
- `#information-extraction`
- `#llm`
- `#rag`
- `#capstone`
- `#mp1`

## Decision Date
August 2026

## Author
Rajees In (rajeesin@adobe.com)

## Reviewers
(To be filled in upon team review)
