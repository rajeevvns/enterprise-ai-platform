# MP2 · Mini-RAG — Ask Your Documents

A small end-to-end RAG pipeline over 5 Sherlock Holmes story summaries:
load → chunk → embed (`text-embedding-3-small`) → store in Qdrant →
retrieve → answer (`gpt-4o-mini`), with citations back to the source
story and section.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI key (and OPENAI_BASE_URL if using a proxy
# such as Vocareum) and your Qdrant URL/API key
source .env
```

## Running

```bash
python mp2_rag.py ingest      # one-time: chunks + embeds the corpus into Qdrant
python mp2_rag.py ask         # interactive Q&A loop
python mp2_rag.py validate    # runs predefined_questions.jsonl + learner_questions.jsonl
```

## Interpreting `validate` output

For each question it prints:
- **Cited vs Expected** — whether the answer's citations include the source
  file the question is keyed to. The `Source-match: N/M` line at the end of
  each batch is the actual pass/fail bar.
- **Facts matched** — how many of the question's `expected_facts` substrings
  appear in the generated answer text. This is diagnostic, not pass/fail: a
  question can have perfect source-match but low fact coverage if the
  retrieved chunks don't happen to contain every fact (see `mp2_reflection.md`
  for a worked example of this).
- **Latency** — wall-clock time for the full retrieve + generate call.

`mp2_validation.txt` in this submission contains the full validation run
(5/5 source-match) plus the complete Q&A transcript (question, answer,
citations) for all 5 questions, so the actual generated answers are visible
without re-running the pipeline.

## Files in this submission

| File | What it is |
|---|---|
| `mp2_rag.py` | Complete pipeline (all 7 TODOs implemented) |
| `data/learner_questions.jsonl` | 3 original questions spanning 3 different stories, at easy/medium/hard (multi-hop) difficulty |
| `mp2_reflection.md` | What worked, what didn't, what I'd change, one surprise |
| `mp2_validation.txt` | Validation run (5/5 source-match) + full Q&A transcript |
| `README.md` | This file |
