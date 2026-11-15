# MP3 Reflection

## Grounding & abstention

The final assistant correctly refused questions about Professor Moriarty, the
boiling point of water, and Sir Charles Baskerville. None of these answers is
supported by the supplied five-story corpus. It answered all six answerable
questions in the final evaluation.

Initially, "What killed Julia Stoner?" incorrectly produced "I don't know based
on the provided documents." The threshold was not responsible: the top retrieved
score was 0.509, above the configured 0.30. The top three chunks described Julia's
death and Helen's predicament but did not identify its cause. The model refused
because its excerpts did not contain the answer, even though the full corpus did.

The swamp-adder evidence appeared at retrieval ranks 7 and 8, with scores of
0.377 and 0.361. Increasing the default retrieval count from 3 to 10 made that
evidence available. Julia's answer then cited the explanation chunk [8], while
the Moriarty question still correctly refused.

The final threshold remains 0.30. Lowering it would not have fixed the missing
evidence, because Julia's query already passed the score gate. No threshold sweep
was performed. Increasing top-K does add context and potential distractors, so
this choice should be revisited against a larger evaluation set.

The Moriarty query also passed the score gate: its highest score was 0.473.
Its refusal therefore depended on the model following the source-only prompt,
not on the threshold alone. Similarity is not proof that an answer is supported.

## Citations

`format_context` assigns each retrieved chunk a number and returns matching
metadata. `validate_citations` extracts numeric markers and separates valid
source numbers from out-of-range numbers.

The offline tests confirmed that [3] is invalid when only two sources exist,
that [0] is invalid, and that [1] is invalid when there are no sources. Repeated
markers are deduplicated, and an answer without valid markers has
`has_citation=False`.

No fabricated citation numbers were observed in the final evaluation. In the
inspected Julia response, [8] pointed to the explanation of the snake bite. The
multi-hop Red-Headed League answer cited [4] for Wilson's copying work and hours
and [3] for Clay's tunnel. The Blue Carbuncle answer cited the excerpt identifying
Peterson's wife and the bird's crop.

These inspected citations supported the requested facts, but the validator
itself only checks number validity. It does not establish that every claim is
entailed by the cited text. The multi-hop response also grouped its citations at
the end; placing a citation immediately after each supported claim would make
manual checking easier.

## Reading your eval table

The saved `mp3_eval.txt` contains nine questions: six answerable and three
should-abstain, including the three added learner questions.

| Metric | Final result | Interpretation |
|---|---|---|
| Retrieval hit rate | 1.0 (6/6) | Every answerable question retrieved its expected story. |
| Grounded & cited rate | 1.0 (6/6) | Every answerable question was answered with a valid citation to its expected story and no invalid markers. |
| Abstention accuracy | 1.0 (3/3) | Every should-abstain question was refused. |
| Hallucination rate | 0.0 (0/3) | No should-abstain question received an answer. |

No metric is numerically weakest on this set. The weakest aspect is the strength
of the evidence behind those numbers: nine questions are a small sample, and
source-level retrieval hits do not prove that the necessary passage was found.
Julia's initial false refusal demonstrated that distinction.

The grounded rate checks whether the expected source was cited, not whether all
expected facts were answered correctly. `expected_facts` provides human guidance
but is not automatically scored. Similarly, this harness's hallucination rate
measures answers to should-abstain questions; it does not count unsupported
claims inside answers to answerable questions.

A useful improvement would be a larger held-out question set containing
paraphrases, ambiguous requests, more multi-hop questions, and difficult
out-of-corpus questions. Passage-level evidence labels and claim-support checks
would strengthen evaluation. Comparing top-K and threshold settings on that set
would help measure false refusals, unsafe answers, and the extra context cost.

## One surprise

An unexpected result was that retrieving the correct story with a high
similarity score was still insufficient to answer a simple question. Julia's
query retrieved the Speckled Band story in all top-three results, but the actual
cause of death was farther down the ranking. Increasing retrieval coverage,
rather than lowering the abstention threshold, fixed the failure.

The Moriarty scores reinforced the lesson from the opposite direction:
apparently relevant Holmes excerpts can score well without containing the
requested character. Retrieval relevance, answerability, and citation support
need to be checked separately.
