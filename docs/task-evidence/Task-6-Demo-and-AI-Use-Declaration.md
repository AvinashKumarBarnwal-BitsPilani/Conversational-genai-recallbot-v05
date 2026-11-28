# Task 6 — Demo Video and AI-Use Declaration

**Project:** RecallBot V05 — e-commerce orders and returns · **Student:** Avinash Kumar Barnwal

<!-- TOC -->
## Table of contents

- [6(a) Demo video](#6a-demo-video)
- [6(b) AI-use declaration](#6b-ai-use-declaration)
  - [Tools used and what for](#tools-used-and-what-for)
  - [A specific case where the AI's suggestion was wrong](#a-specific-case-where-the-ais-suggestion-was-wrong)

<!-- /TOC -->

## 6(a) Demo video

**Video link:** https://drive.google.com/file/d/1PhMvSISE-9Vf2P8JsMvJd4gturEVcDt0/view?usp=drive_link (also in `video_link.txt`; viewable by anyone with the link).

**Length:** We tried to keep the demo video within 5 minutes, but because of the amount of content it had to cover, the video is about 6 minutes 45 seconds long.

The demo is driven by one command, which runs Session A and Session B as two separate processes and then prints the evidence for each item on the checklist:

```powershell
python -m src.dialogue.demo_memory                                          # live run
python -m src.dialogue.demo_memory --verify-only --experiment <experiment>  # replay the evidence of a run
```

| Checklist item (brief, Task 6(a)) | Where it appears in the demo |
|---|---|
| Session A with a confirmed booking | Section 1 — turn 23: `book_slot` runs only after the explicit "yes" and returns a real RMA reference |
| A process restart | "Session A process has EXITED", then section 2 — different PIDs, memory-file SHA-256 hashes identical |
| Session B recalling a semantic fact and an episode | Section 4 — P01 / P03 (facts with `[SM-…]` IDs), P05 / P06 (episodes with `[EP-…]` IDs) |
| A forget request honoured after restart | Section 4 — P04 refuses to reveal the deleted note; "side door" absent from every memory file |
| One knowledge-base answer with its cited chunk | Section 4 — P07, 14-day return window citing `KB-002-C128-0002` |
| One failure case | Section 5 — P11, traced to retrieval and then the reply (Task 4(b)) |
| Walkthrough of the episodic scoring function | Section 6 — score table (recency, importance, relevance) next to `score_episodes()` in `src/memory/episodic_memory.py` |

## 6(b) AI-use declaration

### Tools used and what for

| Tool | Used for |
|---|---|
| **Claude Code** (Anthropic, Claude Opus 5.5, in VS Code) | Days 4–7: Task 3 memory design and implementation (semantic and episodic stores, forget cascade, reflection, context assembler, session runner); Task 4 evaluation scripts (ablation runner, metrics, Qwen judge, error-analysis evidence and the P11/P13 fixes); Task 5 cost, sizing and ANN-benchmark scripts; unit tests; evidence documents, summary figures and README; git commits made only after my review. |
| **ChatGPT** and **OpenAI Codex** | Days 1–3: drafting the synthetic knowledge base, test conversation and probes (Day 1); retrieval code (Day 2); dialogue-pipeline patches and the hang investigation in `References/` (Day 3). |
| *Not assistants — models inside RecallBot* | `qwen3:1.7b` via Ollama (replies, LLM fact extraction, reflection, the LLM judge); `BAAI/bge-small-en-v1.5` (embeddings). |

**What I did myself:** chose and approved every design decision (variant settings, memory design, thresholds, weighting, fixes); ran all experiments locally; reviewed and confirmed every probe mark and every dialogue-quality rating (the assistant proposed them with reasons, I checked each answer against the transcript before setting `judged_by = manual`); reviewed every change before it was committed and merged each pull request. All numbers in the report come from my own runs and logs.

### A specific case where the AI's suggestion was wrong

**What it suggested.** For the 2,048-token context assembler, the assistant wrote the token counter as `len(tokenizer.apply_chat_template(messages, tokenize=True, ...))`, assuming the call returns a list of token IDs.

**How I found out.** In the pilot run (`t3_hm_pilot_A`), the per-turn context-assembly records showed **"2 / 1,856 tokens" on every turn**, while Ollama's own `prompt_eval_count` in `logs/calls.jsonl` reported 463–1,506 tokens for the same calls. With `transformers` 5.x the function returns a `BatchEncoding` (`input_ids`, `attention_mask`), so `len()` counted its two keys. The budget check was therefore not actually enforcing anything; the log confirmed no pilot prompt had exceeded the budget.

**What I changed.** (1) Count `encoded["input_ids"]` instead of the object; (2) added a plausibility test — the templated total must lie between the sum of the components and that sum + 120 tokens — so a silent miscount now fails the test suite; (3) compared the corrected count with Ollama on all 26 pilot Session B calls, found a constant +8 tokens (the empty `<think>` block Ollama inserts), and added it as a measured offset. In the official run the assembler's count matched Ollama on **51 of 51** chat calls. The pilot is kept in the log and labelled as a pilot.

**Two smaller corrections of the same kind:**
- *Answer leakage in a fix.* For the P11 failure, the first query rewrite the assistant proposed added words taken from the expected answer ("returns centre, inspection, refund"). It improved the ranking but would have leaked the answer into the query; I rejected it and kept a generic topic phrase ("next stage of the return and refund process"), noting in the report that it was designed on one probe.
- *Unsafe intent rule.* Under the original forget rule, the probe question "What delivery note did I ask you to forget?" would itself have been classified as a forget request — deleting data in response to a question. This was caught in a code check before memory was wired in, and questions can no longer trigger a forget.
