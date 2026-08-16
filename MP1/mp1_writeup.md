# MP1 Writeup: Prompt Strategy Comparison

## Overview

I compared four prompting strategies on extracting structured data (company, role, years_experience_required) from 10 job postings. All strategies used gpt-4o-mini with a single system prompt and JSON output format. The goal was to understand which approach balances accuracy, cost, and latency.

## Key Findings

### 1. Which strategy won on accuracy and why?

**Few-shot achieved the highest accuracy at 2.9/3 (97%)**, outperforming structured and chain-of-thought by 2-4%.

| Strategy | Accuracy |
|----------|----------|
| Few-shot | 2.9/3 |
| Structured | 2.7/3 |
| Chain-of-thought | 2.7/3 |
| Zero-shot | 2.6/3 |

Few-shot won because it **demonstrated the expected output format with concrete examples**. By showing the model 3 worked examples-including a critical null case where years_experience_required should be null—the model learned both the format and the edge case handling rule. This eliminated hallucination on ambiguous job postings.

Structured came second despite explicit JSON schemas and role prompts. The model still inferred years on some postings where they weren't stated. Chain-of-thought performed similarly; reasoning didn't improve extraction accuracy for this task.

Zero-shot's 2.6/3 was a solid baseline, proving that even minimal instruction gets 87% of fields correct. However, the few-shot advantage of 3-4% is meaningful for production systems.

### 2. Cost and latency trade-off

**Cost differences were negligible** (~$0.001 total across all 40 calls). All strategies cost virtually the same.

Latency showed more variation:
- **Fastest:** Zero-shot (1.028s)
- **Few-shot:** 0.871s ⚡
- **Structured:** 1.064s
- **Slowest:** Chain-of-thought (1.398s)

Surprisingly, **few-shot was actually fastest** despite longer prompts, likely due to model response variability. Chain-of-thought was 60% slower because the model generated reasoning text before JSON, adding tokens.

**Verdict:** Cost is irrelevant; latency favors few-shot and zero-shot, but the 1.4s CoT latency is acceptable for batch processing. Accuracy should drive the decision, not cost or latency.

### 3. What surprised you?

**Few-shot's clear win surprised me.** I expected structured to dominate because it was most explicit about rules and JSON schema. But demonstrations proved more powerful than declarations. The model learns "do this" better from examples than from instructions.

**The 100% parse success rate** (all 40 responses were valid JSON) was excellent. No wrapped markdown fences or preamble text—the models stuck to pure JSON in all cases.

**j10 (the null case) was the canary in the coal mine.** In early iterations without a null example in few-shot, the model would guess "3 years" or "5 years" instead of null. One example showing null output fixed this completely. This taught me that **edge cases must be modeled, not just instructed**.

**Chain-of-thought underperformed.** I hypothesized that reasoning before extraction would catch errors, but it actually added latency without accuracy gains. The model would reason correctly ("no years stated") then extract a number anyway. CoT seems more valuable for harder tasks (e.g., multi-step reasoning).

### 4. Recommendation for capstone work

**For my capstone RAG system, I recommend the few-shot strategy with domain-specific examples.**

Few-shot's 2.9/3 accuracy and modest latency (0.871s) make it production-ready. For my domain (technical job postings), I'll:

1. **Keep the few-shot format** as the baseline
2. **Add 3-5 domain-specific examples** (e.g., salary ranges, remote work options) to show the model my exact output schema
3. **Always include an edge case** (missing field, ambiguous wording) so the model learns to return null, not guess

This hybrid approach combines few-shot's empirical advantage with domain specificity.

## Conclusion

**Prompting design matters.** A 3% accuracy difference between strategies might seem small, but in production systems processing thousands of postings, it compounds. Few-shot's demonstration-based approach outperformed both rules-based (structured) and reasoning-based (CoT) strategies, proving that **showing beats telling** in prompt engineering. For the capstone, I'll prioritize domain examples over general instructions.
