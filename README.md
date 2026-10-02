# Testing a RAG Chatbot with promptfoo

A test framework for an LLM-powered support chatbot that answers from documentation (RAG). It covers the search step, the answer step and the full pipeline, plus robustness against unusual or misleading input.

The system under test is **Bree**, a support assistant for **Brindlework**, a fictional project-management product.

> **Fictional data.** Brindlework, Bree and all documents in `data/corpus/` are invented for testing. None of it describes a real company or product.

## What is tested

The chatbot works in two steps: it **retrieves** relevant chunks from 12 help-center documents, then **generates** an answer from them. Each step is tested on its own and together:

| Layer | Folder | Question it answers |
|---|---|---|
| Retrieval | `eval/suites/01_retrieval` | Does search find the right chunks? |
| Generation | `eval/suites/02_generation` | Given the right chunks, is the answer correct, grounded and well written? (retrieval is skipped; the test supplies the chunks) |
| End to end | `eval/suites/03_end_to_end` | Does the whole chatbot answer correctly, quickly and cheaply? |
| Robustness | `eval/suites/04_robustness` | Does it still work with reworded, misspelled, translated, off-topic or misleading input? |

If a case passes in Generation but fails End to end, the problem is in retrieval. Comparing layers this way locates the root cause.

### Test strategy

```
Test data  →  Test each layer  →  Stress it  →  Grade  →  Triage  →  Fix and re-run
```

1. **Test data:** 66 hand-written test cases across 11 question types (single fact, multi-part, comparison, numbers, dates, follow-ups, unanswerable, ambiguous, conflicting sources, false premise, off-topic). Each case has a reference answer, the chunks that contain the answer, and the key facts.
2. **Test each layer:** retrieval alone, generation alone (with the correct chunks supplied), then the full pipeline.
3. **Stress it:** robustness tests with reworded, misspelled, translated or padded questions, and misleading documents.
4. **Grade:** deterministic checks wherever possible (recall, citations, key facts, latency, cost). An LLM judge is used only where wording varies, and its failing verdicts are reviewed by hand before being trusted.
5. **Triage:** each failure is classified as an app defect, a test-harness defect, a judge defect or a test-data defect before anything is changed.
6. **Fix and re-run:** compare pipeline configurations and prompts on the same cases, adopt the best, and re-run to confirm.

**Test techniques used:**
- **Metamorphic testing:** paraphrases, typos and six languages must give the same answer
- **Negative testing:** unanswerable questions, off-topic requests, false premises
- **Fault injection:** outdated or contradicting documents forced into the context
- **LLM-as-judge** for answers that vary in wording. Judge failures were reviewed by hand, and the context-recall judge was compared once against the deterministic fact-recall metric (40/40 agreement).
- **Flakiness checks:** repeated runs of the same question

## Results

Full run on the recommended configuration (hybrid search + LLM reranking, gpt-6-luna):

| Layer | Suite | Pass |
|---|---|---|
| Retrieval | Ranking metrics (Recall@5 0.92, MRR 0.85) | 47/52 |
| | Context relevance / context recall (LLM-judged) | 43/52 · 36/40 |
| Generation | Correctness · answer relevance · format | 48/50 · 49/50 · 23/23 |
| | Refusal ("I don't know") · clarification | 6/7 · 4/4 |
| | Faithfulness | 40/52 |
| | Hard contexts (buried answer, prompt injection, …) | 6/8 |
| End to end | Answer quality, all 66 cases | 60/66 |
| | Multi-turn · consistency · performance | 5/5 · 15/15 · 16/16 |
| Robustness | Paraphrase · typos · multilingual · long queries | 6/6 · 6/6 · 6/6 · 6/6 |
| | Conflicting docs · distractors · false premises | 6/6 · 5/5 · 4/5 |
| | Off-topic requests | 2/5 |

Typical answer: about 5 seconds and $0.0004. A full run of all suites costs about $0.20.

## Defects found

| Area | Defect |
|---|---|
| App | Follow-up questions ("Can that be extended?") retrieved the wrong chunks |
| App | Two-part questions missed one of the needed facts |
| App | Ambiguous questions were answered with a guess |
| App | Off-topic requests are answered (e.g. it tells jokes) |
| Test harness | Latency was overstated about 4× because promptfoo queued requests behind one worker |
| Test harness | Splitting chunk text broke markdown tables |
| Judge | promptfoo's `context-recall` judge failed to grade 25 of 40 cases (output format errors) |
| Test data | Correct "contact support" answers were failed because reference answers didn't allow it |

## Running the tests

Requirements: Python 3.11+, Node.js 22.22+ (needed by promptfoo), an OpenAI API key.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
npm install
copy .env.example .env                  # add your OPENAI_API_KEY

python -m rag_app.cli ingest            # build the search index
```

Run tests with promptfoo from the project folder, with the virtual environment activated (promptfoo uses its Python):

```powershell
npx promptfoo eval -c eval/suites/00_smoke/answer_smoke.yaml                          # one suite
npx promptfoo eval -c eval/suites/02_generation/refusal.yaml --filter-metadata id=un-002   # one case
npx promptfoo eval -c eval/suites/02_generation/correctness.yaml --filter-first-n 5        # first 5 tests
npx promptfoo view                                                                    # browse results in the browser

npx promptfoo eval -c eval/suites/01_retrieval/ranking_metrics.yaml -o eval/results/ranking_metrics.json
python eval/scripts/summarize_results.py --failures   # summary of the saved results
pytest                                                # unit tests for the app
```

promptfoo reads the API key from `.env`. If `OPENAI_API_KEY` is also set in your system environment, that value is used instead.

Try the chatbot directly: `python -m rag_app.cli chat --sources`

## Project layout

```
rag_app/          the chatbot (system under test)
configs/          pipeline configurations; recommended.yaml is the tested setup
data/corpus/      the 12 fictional help-center documents
eval/
  suites/         promptfoo test suites, by layer
  providers/      connect promptfoo to the chatbot (full pipeline, retrieval only, generation only)
  asserts/        custom metrics (Recall@k, MRR, fact recall, citation check)
  datasets/       master test cases and the data-file half of each suite
  scripts/        summarize_results.py
```
