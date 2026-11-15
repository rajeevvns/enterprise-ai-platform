# MP3 - Week 12: Grounded Q&A

A self-contained working copy of the Week 12 learner starter. The original
course bundle, MP2, and capstone are unchanged. The four required functions are
implemented, and the nine-question live evaluation, learner questions,
evidence-based reflection, and narrated demo are included.

## Setup

Run from the repository root:

```bash
cd mini-projects/mp03-week12-grounded-qa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your OpenAI and Qdrant credentials, then run `source .env`.
Never commit `.env`; it is ignored by the repository. Use OpenAI
`gpt-4o-mini`, `text-embedding-3-small`, and Qdrant directly, without frameworks.
The starter uses the dedicated collection `mp3_sherlock`; verify that this name
does not belong to another project before ingesting.

The working copy uses `qdrant-client` 1.20.x for the Qdrant Cloud server 1.20.x
and retrieves through `query_points`. The original starter's older client pin
and `search` call have been updated together; compatibility checking stays enabled.

## Work sequence

1. Review `format_context`, `validate_citations`, `should_abstain`, and
   `evaluate` in `mp3_grounded_qa.py`.
2. Run `env -u OPENAI_API_KEY -u QDRANT_URL -u QDRANT_API_KEY python mp3_grounded_qa.py selftest`.
   This runs the synthetic checks without initializing live clients.
3. Run `python mp3_grounded_qa.py ingest`, then `python mp3_grounded_qa.py ask`.
   Check both cited answers and correct refusals.
4. Review `data/learner_questions.jsonl`: two answerable
   questions across different stories (one multi-hop), plus one unanswerable.
5. Run `python mp3_grounded_qa.py eval > mp3_eval.txt` and inspect the output.
   This regenerates the included evaluation artifact using live service results.
6. Review `mp3_reflection.md` and the demos before submission.

See `MP3_FAQ.md` for function contracts, evaluation metrics, and troubleshooting.

## Demo

[`demo/mp3_demo.mp4`](demo/mp3_demo.mp4) is a 2-minute 30-second narrated replay of
actual captured terminal output: offline selftest, Julia's cited answer, the
multi-hop Red-Headed League answer, Moriarty abstention, and the saved evaluation
table. It is not a live desktop screen recording; narration is computer-generated.
The [transcript](demo/mp3_demo_transcript.txt) includes narration and captured output.

Review the reflection and demo before submission so they represent your own
understanding. If your course requires your own voice or a live screen recording,
use the transcript as a recording guide instead.

## Submission

The ready-to-upload archive is
[`Week 12_Graded Mini Project_SINGH.zip`](submission/Week%2012_Graded%20Mini%20Project_SINGH.zip).
It includes the required deliverables, supporting files, and
`MP3_Week12_demo.mov`: the supplied screen recording compressed and sped up
to 1.12x (2:57), preserving all content. The 358 MB original stays local and is
ignored by Git; the approved compressed copy is inside the ZIP.

Submit the implemented script, `mp3_eval.txt`, your learner questions,
`mp3_reflection.md`, and the demo. Keep the corpus, dependencies, and environment
template with the code so it can be run independently.

The assignment requires a personal GitHub repository separate from the
capstone. Develop here, then publish this folder's contents to a separate MP3
repository for submission, excluding `.env`, `.venv`, and Python caches.
