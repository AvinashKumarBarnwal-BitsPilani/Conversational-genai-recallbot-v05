# Task 1 – Test Data and Probes

## 1. Requirement from the mini project

Task 1 requires three primary data assets:

1. `data/test_conversation.json` – at least 40 user turns split into Session A (turns 1–25) and Session B (turns 26 onward), with the specified user facts, updates, forget request, episodes, low-importance turns, slot-filling workflow, state follow-ups, clarification and knowledge-base questions.
2. `data/kb_questions.json` – at least 25 retrieval questions with a gold answer span and document ID, including at least five exact-code/ID questions, at least five paraphrased questions that share no keyword with the answer span, and at least two questions whose answers are available only in a newer document version.
3. `data/probes.json` – at least 14 probes with expected answers covering all seven categories (R, U, F, E, K, N, S), plus at least two probes for the assigned variant twist.

The assigned variant is **V05 – E-commerce orders and returns**, with **Twist E: Small context window**, a **2,048-token context window**, and **trim window k = 3 turns**.

## 2. Implemented data assets

### 2.1 Knowledge base supporting the test data

| Check | Implemented value | Evidence |
|---|---:|---|
| Knowledge-base documents | 16 | `data/kb/manifest.json` |
| Total word count | 6,579 | `data/day1_manifest.json`; `python -m src.main --check` |
| Required word range | 4,000–8,000 | Day 1 validation |
| Business codes and IDs | At least 10; multiple order, SKU, return, payment, shipping and error identifiers | `data/kb/*.md`; `test_kb_has_required_volume_codes_and_two_version_pairs` |
| Old/new returns-policy pair | KB-001 (2025, superseded) and KB-002 (2026, active) | `data/kb/manifest.json` |
| Old/new shipping-policy pair | KB-016 (2025, superseded) and KB-005 (2026, active) | `data/kb/manifest.json` |

The KB contains synthetic ShopSphere policies, FAQs and service notes. The manifest records document IDs, file names, titles, effective dates, status and version-control information.

Examples of domain identifiers include `RET-E14`, `ERR-RMA-04`, `RMA-BOOK`, `ORD-74021` and `SKU-AUR-220`.

### 2.2 Scripted conversation

`data/test_conversation.json` contains **40 user turns** divided as follows:

| Session | Turn range | User turns |
|---|---:|---:|
| Session A | 1–25 | 25 |
| Session B | 26–40 | 15 |
| **Total** | **1–40** | **40** |

The conversation has nine planted user-fact records. The fact records include a key, value, source turn, update reference and status.

| Fact / preference | Source turn | Update / deletion | Expected final status |
|---|---:|---:|---|
| Preferred name: Maya | 2 | — | Active |
| Notification channel: email | 4 | Updated at turn 20 | Superseded |
| Preferred city: Bengaluru | 5 | — | Active |
| Delivery note: side door when carrier permits | 6 | Forget request at turn 24 | Deleted |
| Delivery window: 18:00–21:00 | 7 | Updated at turn 19 | Superseded |
| Return method: home pickup | 8 | — | Active |
| Packaging preference: recyclable where available | 9 | — | Active |
| Delivery window: 19:00–21:00 | 19 | — | Active |
| Notification channel: SMS | 20 | — | Active |

This covers more than the required eight user facts, two updates that override earlier facts, and one explicit forget request.

### 2.3 Events and low-importance turns

The scripted conversation identifies four expected events for episodic memory:

| Episode ID | Source turn | Event |
|---|---:|---|
| EP-000 | 6 | Delivery-preference event; later subject to the forget request |
| EP-001 | 10 | Late-delivery complaint for order `ORD-88216` |
| EP-002 | 17 | Decision to exchange the USB-C charger in order `ORD-66310` rather than request a refund |
| EP-003 | 23 | Return-pickup booking scenario for earbuds in order `ORD-74021` |

Seven turns are explicitly marked low-importance: **1, 3, 11, 25, 26, 38 and 40**. These provide filler/small-talk examples that should not create durable semantic facts.

### 2.4 Slot-filling workflow and dialogue-state cases

The scripted return-pickup task is represented in Session A:

| Turn | Scripted action |
|---:|---|
| 12 | Start return request; read order and identify item; ask for reason. |
| 13 | Capture the defective-item reason. |
| 14 | Capture pickup date: 28 November 2026. |
| 15 | Resolve the follow-up “And for Saturday?” using the active task state and collect the after-6-pm window. |
| 22 | Present the complete pickup summary and request explicit confirmation. |
| 23 | User confirms; the expected next action is the database write after slot revalidation. |

The script also contains state-follow-up turns (including turns 15, 22 and 33) and a clarification case at turn 37, where the user asks to book a pickup for “the other item” without identifying the order or item.

**Execution evidence boundary:** The conversation fixture defines the confirmation point and expected database-write action. The actual write, returned booking/RMA reference and tool-call log must be demonstrated by the running dialogue pipeline under Task 2; this Task 1 data check alone does not claim that a live database write has already occurred.

### 2.5 Knowledge-base questions

`data/kb_questions.json` contains **32 retrieval questions**, each with a `question_id`, question text, `gold_answer_span`, expected answer and source `document_id`.

| Required category | Implemented count | Requirement |
|---|---:|---:|
| Exact code or ID questions | 10 | At least 5 |
| Paraphrased questions with no answer keyword | 5 | At least 5 |
| Version-sensitive questions | 7 | At least 2 |
| Questions tagged `new_version_only` | 2 | At least 2 |


**Version-specific content audit:** Q030 asks about the 23:59 local-time cutoff introduced in the current returns policy (KB-002), which is absent from the archived 2025 returns policy (KB-001).

Q032 asks what ShopSphere should offer when the preferred delivery window is unavailable. Its gold answer is sourced from the current shipping policy (KB-005), and this rule is absent from the archived 2025 shipping policy (KB-016).

Both questions therefore provide examples of answers available only in the newer document versions.


Knowledge-base questions are included in both sessions of the scripted conversation. Session A includes KB questions at turns 16, 18, 21 and 22; Session B includes KB questions at turns 31, 33, 34 and 35.

### 2.6 Probe set

`data/probes.json` contains **16 probes**. It covers all seven required categories and includes two probes for Twist E.

| Probe category | Meaning in the mini project | Count |
|---|---|---:|
| R | Semantic recall | 2 |
| U | Updated fact | 1 |
| F | Forgotten fact | 1 |
| E | Episodic recall | 3 |
| K | Knowledge-base answer | 3 |
| N | Not available / must not invent | 2 |
| S | State follow-up | 2 |
| Twist E | Small-context-window probes | 2 |
| **Total** |  | **16** |

The two Twist E probes check (1) enforcement of the 2,048-token context budget, including prioritisation and recording what is cut, and (2) retrieval of the latest delivery preference even when the fact is older than the last three raw turns.

The five surprise probes released by the professor on 24 November are a separate Task 4(c) input. `data/surprise_probes.json` is intentionally a pending-release placeholder at this stage; no surprise probes have been invented or pre-populated.

## 3. Validation and test evidence

The following checks were run against the data assets:

| Validation | Evidence / test | Result |
|---|---|---|
| KB document count and word count | `test_kb_document_and_word_count`; full Task 1 coverage test | Passed |
| Gold answer spans exist in their labelled source documents | `test_kb_gold_spans_exist_in_source`; `test_kb_questions_have_codes_paraphrases_and_new_version_cases` | Passed |
| Conversation turn count and session boundaries | `test_conversation_structure_and_facts` | Passed |
| Probe categories and two Twist E probes | `test_probe_coverage_including_small_context_twist`; `test_probes_cover_seven_categories_and_two_small_context_twists` | Passed |
| Full Task 1 coverage | `tests/test_task1_full_coverage.py` | Passed |
| Project data check | `python -m src.main --check` | Passed |

The reported `python -m src.main --check` output confirmed: 16 KB documents, 6,579 KB words, 40 user turns (25 in Session A and 15 in Session B), 32 KB questions, 16 probes and two twist probes. It ended with `CHECK PASSED: Day 1 data minimums satisfied.`

The final combined test run reported **26 passed**. This includes the Day 1 data tests and the full Task 1 coverage tests; the full test-suite result is also recorded in the project validation history.

## 4. Task 1 completion summary

The three required Task 1 data assets are prepared and validated against the stated minimums:

- `data/test_conversation.json` – 40 turns, Session A/B split, nine planted facts, two superseding updates, a forget request, four expected episodes, seven low-importance turns, a slot-filling scenario, state follow-ups, clarification and KB questions across both sessions.
- `data/kb_questions.json` – 32 questions with gold spans and source document IDs, including 10 exact-code/ID, five paraphrased-no-answer-keyword and two `new_version_only` questions.
- `data/probes.json` – 16 probes covering R, U, F, E, K, N and S, plus two small-context Twist E probes.


**Status: Task 1 data assets meet the required coverage criteria.** Q032 has been corrected to use a question whose gold answer is exclusive to the current shipping policy (KB-005), alongside Q030 for the current returns policy (KB-002). The corrected question set has been validated against the source documents and the retrieval evaluation has been regenerated.

## 5. Reproduction commands

Run these commands from the project root with the project virtual environment activated:

```powershell
python -m src.main --check
python -m pytest tests/test_day1_data.py -v
python -m pytest tests/test_task1_full_coverage.py -v
```

For the full project suite:

```powershell
python -m pytest -v
```

## 6. Evidence files

- `data/day1_manifest.json` – variant settings and aggregate asset counts.
- `data/kb/manifest.json` – KB document IDs, versions, effective dates and statuses.
- `data/test_conversation.json` – 40-turn scripted conversation, fact inventory, expected episodes and low-importance turn list.
- `data/kb_questions.json` – retrieval questions and gold answer spans.
- `data/probes.json` – expected probe cases and Twist E probes.
- `tests/test_day1_data.py` and `tests/test_task1_full_coverage.py` – automated coverage checks.
