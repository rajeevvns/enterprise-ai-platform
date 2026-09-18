# MP2 Reflection

## What worked

Paragraph-boundary chunking with heuristic section-header detection (short lines without terminal punctuation, e.g. "The Strange Work") mapped cleanly onto how these story summaries are actually written — each story naturally broke into 13-18 chunks of 300-600 characters along its own scene breaks, with no manual tuning needed per file. The bigger win, though, was embedding the section title as part of the chunk text itself rather than keeping it purely as citation metadata. For one query ("who was the assistant... and what was his real identity"), the correct chunk — titled "The Identity of the Assistant" — went from dead last (score 0.423 out of 13 chunks, even at k=8) to rank 4 (0.540) once its own title's keywords ("identity", "assistant") were folded into the embedded text instead of being discarded after chunking.

## What didn't work

Two of my five validation questions had low `expected_facts` coverage despite passing the source-match check, and they turned out to be two genuinely different failure modes. The pawnshop-assistant question failed because the answer chunk described "Spaulding"/"Clay" by name with no literal overlap with the query's words ("assistant", "pawnshop") — a case where dense embedding similarity rewarded surface vocabulary overlap over the chunk that actually contained the answer. The blue-carbuncle "wrong goose" question failed for a completely different reason: it's a real multi-hop question needing two separate chunks ("James Ryder Confesses" and "Tracing Henry Baker"), and both scored just below the k=3 cutoff (0.479 and 0.470, versus 0.490 for the 3rd slot) — not a relevance problem, just not enough room in a fixed top-3 for a question that needs two facts from two different places.

## What I'd change

With another 5 hours I'd try the Stretch hybrid+rerank path specifically to see whether it helps the multi-hop case — BM25 keyword matching plus a cross-encoder rerank might resurface both "James Ryder Confesses" and "Tracing Henry Baker" together, since a cross-encoder scores each candidate against the full question rather than a single query embedding. I'd also experiment with per-query adaptive k (retrieve more candidates, then let the cross-encoder narrow it back down) rather than a fixed k=3 for every question regardless of whether it's single-hop or multi-hop.

## One surprise

I expected "bad chunking" to be the main failure mode going in (that's what the brief warns about), and chunking itself was fine — the surprise was that the retrieval layer has its own distinct, and non-overlapping, failure modes even with clean chunks: one is a vocabulary/embedding mismatch (fixable by putting more signal into the embedded text), and the other is a capacity limit inherent to a fixed k on multi-hop questions (not really fixable by touching the embedding at all — it needs either a bigger k or a fundamentally different retrieval strategy). Two different-looking low scores in the validation output turned out to need two completely different diagnoses.
