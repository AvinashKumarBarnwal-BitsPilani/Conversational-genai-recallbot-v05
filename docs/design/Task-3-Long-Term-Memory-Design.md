# Task 3 (c–f) — Long-Term Memory and Context Engineering: Design

**Status:** Iteration 1 design, before implementation. Day 4, 24 November 2026.
**Scope:** Task 3(c) semantic memory, 3(d) episodic memory and reflection, 3(e) context engineering, 3(f) Twist E.
**Out of scope:** Task 3(a)/(b) retrieval is complete (Day 2) and is reused unchanged.

---

## 1. Audit findings that shape this design

| # | Finding in the current code | Consequence for Task 3 |
|---|---|---|
| A1 | `DialogueManager` memory branches (`forget_request`, `semantic_memory_query`, `episodic_memory_query`, `provide_user_fact`, `update_user_fact`) return "not connected" placeholders; `report_complaint` / `report_decision` have no handler. | Memory must be wired into these handlers; episodes are written from them. |
| A2 | `memory/` and `src/memory/` are empty. | Everything below is new code. |
| A3 | Rule-based NLU extracts facts for turns 2, 4, 5, 6, 7, 9, 19, 20 but **nothing for turn 8** ("a pickup from my home"); turns 7/19 produce a raw `time_window`, not a fact key. | A dedicated extractor maps entities to canonical keys and uses a Qwen fallback for missed fact-like turns. |
| A4 | `QwenNLG._build_messages` trims only KB evidence; it sends the full workflow JSON every turn and never trims history. Task 2 had 24 budget fallbacks. | A context assembler replaces the evidence-only trimming and records every dropped item. |
| A5 | `replay_session` rejects Session B and `DialogueManager` accepts only `full/trim/summary`. | A new session runner and the Task 4 configs are added. |
| A6 | The scripted Session A never books the pickup: turn 15 offers two slots, the script never chooses, turn 23 has nothing pending. | Booking fix (Section 9); required for EP-003, probe P05 and the demo. |
| A7 | `retrieved_tokens` on chat calls counts KB evidence only. | Memory tokens placed in the prompt are added. |

## 2. Architecture

```text
User turn ─► NLU (analyze_utterance) ─► DialogueStateTracker ─► DialogueManager
                                                                  │
            ┌─────────────────────────────────────────────────────┼───────────────────────────┐
            ▼                         ▼                           ▼                           ▼
  FactExtractor (rules → Qwen)   EpisodeWriter + Importance   search_kb / get_order /     MemoryManager.forget()
            │                         │                       check_pickup_slots / book_slot      │
            ▼                         ▼                                                           │
  SemanticMemoryStore           EpisodicMemoryStream  ◄──────── cascade redaction ────────────────┘
  (records + vector index)      (stream + vector index, reflections)
            │                         │
            └──────────► MemoryRetriever ◄──┘
                              │ facts (cosine) / episodes (recency+importance+relevance)
                              ▼
                     ContextAssembler (2,048 budget, Qwen tokenizer, drop log)
                              ▼
                     QwenNLG.generate_reply ─► reply + canonical citations
Session end ─► ReflectionWriter (Qwen) ─► EpisodicMemoryStream ─► MemoryManager.save() ─► disk
```

Semantic and episodic memory stay separate. A preferred delivery window is a **semantic fact**. The turn where the customer changed it is an **episode** that links to the fact by `memory_id`.

## 3. Module layout (new code only; existing modules are extended, not duplicated)

| Path | Responsibility |
|---|---|
| `src/memory/records.py` | `SemanticFact`, `Episode` dataclasses; JSON (de)serialisation; ID generation |
| `src/memory/clock.py` | Logical session clock used for timestamps and recency (Section 6.2) |
| `src/memory/vector_store.py` | Small persisted flat cosine index keyed by `memory_id`, with **delete** (NumPy; same pattern as `DenseIndex`) |
| `src/memory/semantic_memory.py` | Keyed records: add, supersede, active-only lookup, forget; owns its vector index |
| `src/memory/fact_extractor.py` | Rules first, Qwen JSON fallback; logs `call_type=extract` |
| `src/memory/episodic_memory.py` | Memory stream, rule-based importance (`call_type=importance`), scored retrieval, redaction |
| `src/memory/reflection.py` | End-of-session reflection via Qwen (`call_type=reflect`) |
| `src/memory/memory_manager.py` | Facade: per-run memory directory, load/save, forget cascade, config switches |
| `src/context/context_assembler.py` | Budgeted prompt assembly, component token table, drop log |
| `src/dialogue/run_session.py` | One session per process: `--session A/B --config ... --experiment ...` |

## 4. Configs (designed now for Task 4)

| Config | Short-term history | KB (`search_kb`) | Semantic memory | Episodic memory + reflection |
|---|---|---|---|---|
| `no_rag` | trim k=3 | – | – | – |
| `hybrid` | trim k=3 | ✓ | – | – |
| `hybrid_sem` | trim k=3 | ✓ | ✓ | – |
| `hybrid_mem` | trim k=3 | ✓ | ✓ | ✓ |

- Short-term history uses V05 trim (k=3) in all Task 4 configs. The state tracker is always on.
- A disabled memory type is **neither written nor read**, so its extract, embed, importance and reflect calls are not charged to that config's token totals.
- Task 2 configs (`full`, `trim`, `summary`) keep their existing behaviour.

## 5. Semantic memory — Task 3(c)

### 5.1 Record schema (`memory/runs/<experiment>/semantic_facts.json`)

```json
{
  "memory_id": "SM-0004",
  "user": "CUST-MAYA-1042",
  "key": "preferred_delivery_window",
  "value": "19:00-21:00",
  "source_turn": 19,
  "session": "A",
  "date": "2026-11-21T10:38:00+05:30",
  "status": "active",
  "supersedes": "SM-0003",
  "superseded_by": null,
  "extraction_method": "rule",
  "rule_id": "FX-TIME-WINDOW"
}
```

- **Canonical keys** match `planted_user_facts` exactly: `preferred_name`, `notification_channel`, `preferred_city`, `delivery_note`, `preferred_delivery_window`, `return_method_preference`, `packaging_preference`. Qwen may introduce a new snake_case key only if none of these fits; the key is validated by regex.
- **Statuses:** `active` → `superseded` (on update). Forgotten facts are **removed**, not given a status, so their value never stays on disk.
- **Update:** a new value for an existing active `(user, key)` marks the old record `superseded`, sets the links, and removes the old vector from the index, so a stale fact cannot be retrieved. A repeat of the same value is a no-op.
- **Vector index:** each active fact is embedded with the project's BGE embedder as `"<key in words>: <value>"`, logged as `call_type=embed`. The index stores only active facts.
- **Retrieval:** cosine similarity over active facts, top-k=3, minimum similarity θ_sem (tuned once on Session A and then fixed). An exact key match from NLU (for example "delivery window") is ranked first.

### 5.2 Extraction (rules first, Qwen fallback)

1. **Rule pass** (`model="rule-based"`, logged as `extract`): maps NLU entities to canonical keys. For example, `time_window` on a fact turn becomes `preferred_delivery_window`; "pickup from my home" or "home pickup" becomes `return_method_preference`.
2. **Fallback trigger:** the turn is *fact-like* (NLU intent is `provide_user_fact` / `update_user_fact`, or a first-person preference cue such as "I prefer", "my … is", "I usually", "please always") **and** the rule pass found nothing.
3. **Qwen fallback** (`call_type=extract`, temperature 0, JSON-only) returns `[{key, value}]`, which is then validated: the key must be canonical or valid snake_case, and the value must be non-empty, ≤80 characters and appear in or be derivable from the utterance. Invalid output is discarded and logged with `status=rejected`.
4. **Filler turns never reach extraction:** greeting, small_talk, closing and confirm intents skip it, which satisfies "filler produces no record".
5. **Never stored:** passwords, OTPs or card numbers (rule guard).

**Precision and recall** are computed against all 9 `planted_user_facts`, including the superseded and later-deleted ones, matched on `(key, normalised value, source_turn)`. Reported for rules-only vs rules+fallback, from logged runs only.

## 6. Episodic memory — Task 3(d)

### 6.1 Episode schema (`memory/runs/<experiment>/episodes.json`)

```json
{
  "episode_id": "EP-0007",
  "user": "CUST-MAYA-1042",
  "session": "A",
  "timestamp": "2026-11-21T10:46:00+05:30",
  "source_turn": 23,
  "type": "booking",
  "description": "Return pickup booked for SKU-AUR-220 (ORD-74021) on 2026-11-28, 18:00-20:00, slot PICK-20261128-E1, RMA RMA-...",
  "entities": {"order_id": "ORD-74021", "sku": "SKU-AUR-220", "rma_id": "RMA-..."},
  "importance": 9,
  "importance_method": "rule",
  "linked_memory_ids": [],
  "redacted": false
}
```

### 6.2 What creates an episode, and its importance (rule-based, logged as `importance`)

| Event (DM handler) | Type | Importance |
|---|---|---:|
| Confirmed booking with a real RMA from `book_slot` | `booking` | 9 |
| Complaint (turn 10) — order read, customer-reported; **no invented case ID** | `complaint` | 8 |
| Decision (turn 17) | `decision` | 7 |
| Return request started (turn 12) | `task_event` | 5 |
| Fact updated (turns 19, 20) | `preference_update` | 4 |
| Fact first saved (turns 2–9) | `preference_event` | 3 |
| End-of-session reflection | `reflection` | 6 |
| Small talk, thanks, greeting | — (not written; below threshold 3) | 1 |

**Episode text never contains a value that has already been forgotten.** The forget request itself writes a `forget_event` episode that records only the key ("customer asked to forget a saved delivery note"), never the value.

### 6.3 Retrieval score

`score = w_r·R̂ + w_i·Î + w_s·Ŝ`. Each component is min-max normalised across the candidate set, as in Generative Agents.

- **Recency:** `R = 0.995^(hours since the episode)`, the Generative Agents decay. Time comes from a **logical session clock**: Session A starts 2026-11-21 10:00 and advances 2 minutes per turn; Session B starts 2026-11-24 10:00. Wall-clock time cannot be used, because a scripted replay finishes in minutes and recency would be about 1.0 for every episode.
- **Importance:** `I = importance / 10`.
- **Relevance:** `S` = cosine(query, episode) using the same BGE model.
- **Weightings compared:** A relevance-focused (0.20, 0.30, 0.50); B recency-focused (0.40, 0.30, 0.30); C equal (⅓, ⅓, ⅓). All are candidates; the one kept is chosen on the E probes (P05, P06, P13) **plus** `data/episodic_queries.json` (about 12 queries with gold episode types and source turns, including "most recent" queries where recency should matter). Metrics: hit@1 and MRR.
- Top-k = 2 episodes plus at most 1 reflection, with a minimum relevance θ_epi.

### 6.4 Reflection

At the end of each session, one Qwen call (`call_type=reflect`, temperature 0) receives that session's episodes with importance ≥ 4 and the active facts. It returns 1–3 insight sentences, stored as `type=reflection` with `linked_memory_ids` pointing to the source episodes. Session B retrieves the Session A reflection through the same scoring.

## 7. Forget cascade (turn 24) — persistent across restart

`MemoryManager.forget(user, key)`:
1. Deletes **every** record of that key (active and superseded) from `semantic_facts.json` and removes their vectors.
2. Deletes episodes of type `preference_event` / `preference_update` that link to those records (EP-000).
3. For other episodes and reflections whose text or entities contain any forgotten surface form: redacts the text to `[forgotten <key in words>]`, sets `redacted: true` and **re-embeds** the episode (the old vector is deleted).
4. Writes a tombstone `{key, value_sha256, forgotten_at, source_turn}`. **No plaintext is kept**; the hash lets a test prove the value is absent without storing it.
5. Within the live process: registers the value for redaction in short-term history and summary (existing `RollingSummaryMemory.add_redaction`; extended to full and trim).
6. Saves atomically to disk.

**Acceptance test:** forget, then restart in a new process, then grep every file under the memory directory and every vector-index manifest for the value. Probe P04 must return the "no longer retained" answer.

## 8. Context assembly — Task 3(e) and Twist E

**Budget:** 2,048 context − 160 output reserve − 32 safety = **1,856 input tokens**, counted with the Qwen3 tokenizer (`Qwen/Qwen3-1.7B`, local cache), never the BGE tokenizer. Each message adds a fixed chat-template overhead, measured once against Ollama's `prompt_eval_count`.

| Priority | Component | Memory type | Home | Rule |
|---:|---|---|---|---|
| 1 | System prompt (persona, scope, grounding rules, fallback sentence) | Procedural | External file `skills/` → prompt | Mandatory |
| 2 | Current task: user message, compact dialogue state, DM draft | Working | Model-native | Mandatory |
| 3 | Instructions: **current workflow step only**, not the whole JSON | Procedural | External file → prompt | Mandatory when a task is active |
| 4 | Tools: compact definitions of `search_kb`, `get_order`, `check_pickup_slots`, `book_slot` | Procedural | Prompt | Mandatory, about 60 tokens |
| 5 | Long-term memory: semantic facts (SM-ids), episodes and reflection (EP-ids) | Semantic / Episodic | External store → prompt | Whole items, best first |
| 6 | Retrieved knowledge: KB chunks (chunk IDs) | Semantic (domain) | External index → prompt | Whole items, then truncate the last one |
| 7 | Conversation history: last k=3 exchanges | Short-term | Model-native | Oldest exchange dropped first |

- **Intent-aware ordering:** for KB intents, priorities 5 and 6 swap; for memory queries, memory comes first.
- **Drop log:** every turn records `included[]` and `dropped[]` (id, component, tokens, reason). These go into the chat call's `extra.context_assembly` in `calls.jsonl` and into `results/task3_context_assembly.json`. This is the evidence that shows what gets cut.
- **Grounding rules (in the system prompt):** answer customer facts only from SM records, events only from EP records, and policy only from KB chunks with an active status. Cite the ID of every item used, for example `[SM-0004]`, `[EP-0007]`, `[KB-002-C128-0003]`.
- **Exact fallback sentence**, used when no memory or chunk clears its threshold:
  > `I don't have that information in my saved details or the ShopSphere knowledge base, so I won't guess.`

  It is emitted by the DM deterministically, not left to the model, and tested.

## 9. Booking fix (decision approved 24 Nov)

When the customer asks for the **summary to confirm** (`request_task_summary`), the requested window is **open-ended** (for example `after 18:00`), and several slots match, RecallBot proposes the **earliest slot that satisfies the constraint**, names the alternatives, and places that proposal in `pending_confirmation`. The write still requires the explicit "yes" at turn 23, and every existing guard is unchanged: ownership, SKU, live revalidation and duplicate prevention. Turn 15 still lists both options (probe P10 behaviour is unchanged). Bounded or ambiguous windows still require an explicit choice.

## 10. Sessions as separate processes

```powershell
python -m src.dialogue.run_session --session A --config hybrid_mem --experiment t3_hm_01
python -m src.dialogue.run_session --session B --config hybrid_mem --experiment t3_hm_01
```

- **Memory directory:** `memory/runs/<experiment>/`. Session A refuses to start if the directory already exists. Session B refuses unless Session A's manifest says `complete`.
- **Run IDs:** `<experiment>_A` and `<experiment>_B` (one per process, as the logger requires).
- **Proof of separation:** `session_manifest.json` records each session's PID, start and end time, and the SHA-256 of every memory file saved by A and loaded by B. B's loaded hashes must equal A's saved hashes, and the PIDs must differ.
- **Deliverable:** after the final `hybrid_mem` Session B, the run's files are copied to the top level of `memory/`.

## 11. Logging additions

| call_type | Where | Tokens |
|---|---|---|
| `extract` | rule pass (`rule-based`, input tokens of the utterance) and Qwen fallback (Ollama counts) | as measured |
| `importance` | rule-based scorer | input tokens of the episode text, 0 output |
| `embed` | fact and episode vectors, query vectors for memory search | existing embedder accounting |
| `reflect` | session-end reflection | Ollama counts |
| `chat` | `retrieved_tokens` = KB + semantic + episodic tokens actually placed in the prompt | |

## 12. Acceptance criteria

| ID | Criterion | Evidence |
|---|---|---|
| AC-1 | 9 planted facts: precision and recall reported (rules-only and rules+Qwen); filler turns 1, 3, 11, 25 produce 0 records | `results/task3_semantic_memory_metrics.json`, unit tests |
| AC-2 | After Session A: `preferred_delivery_window`=19:00-21:00 and `notification_channel`=SMS active; old values superseded and absent from the index | memory files, tests |
| AC-3 | `delivery_note` absent from facts, index, episodes and reflection **after restart** (hash check) | restart test, Session B P04 |
| AC-4 | Episodes EP for turns 10, 17 and 23 (with a real RMA) present with importance; weightings A/B/C compared; chosen weighting justified | `results/task3_episodic_weight_comparison.csv` |
| AC-5 | At least one Session A reflection stored and retrieved in Session B | Session B transcript plus chat-call context log |
| AC-6 | Every chat prompt ≤ 1,856 input tokens by the Qwen tokenizer; drop log present; one Session B turn shown as a full 3(e) table | `results/task3_context_assembly.json` |
| AC-7 | Fallback sentence emitted verbatim for P09 | Session B transcript, unit test |
| AC-8 | Session B in a separate process: different PID, matching file hashes | `session_manifest.json` |
| AC-9 | Twist probes P15 and P16 reported, passing or failing, with explanation | Task 3 evidence doc |
| AC-10 | `python -m pytest tests/` passes; `calls.jsonl` only appended to | test run |

## 13. Iteration plan (Day 4)

| Iteration | Content | Live LLM? |
|---|---|---|
| 1 | This design, plus the booking fix and its tests | No |
| 2 | records, clock, vector store, semantic store, extractor, DM fact/update handlers, P/R harness | No (fakes) |
| 3 | Episodic stream, importance, scoring, DM complaint/decision/booking episodes, episodic query set | No (fakes) |
| 4 | Reflection, memory manager (persist/load), forget cascade, restart tests | No (fakes) |
| 5 | Context assembler in NLG, configs, memory-query handlers and fallback, `run_session`; **checkpoint**, then live Session A/B `hybrid_mem` | **Yes** |
| 6 | Weighting comparison and P/R from logged runs, twist probes, evidence doc and README | Yes (evaluation) |
