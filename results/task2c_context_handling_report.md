# Task 2(c): Final context-handling analysis

## Scope and official results

Three independent corrected Session A executions processed all 25 original turns. Run IDs are `task2c_full_snapshot1`, `task2c_trim_snapshot1`, and `task2c_summary_snapshot1`; all reset fingerprints match (`b839ab6f76c491f120b796b9189c03ac067b90b7dd027d28d05bc3be8ac2abb2`). Restorable snapshots are under `results/task2c_snapshots/`. The separate Task 2(c) manifest is `results/task2c_session_A_run_manifest.json`. The historical Task 2(b) transcripts and manifest were not rewritten.

The only official Session A probes in `data/probes.json` are P08 (turn 18, KB question about ERR-RMA-04) and P10 (turn 15, state follow-up for Saturday after 6 pm). The reviewed six-row result table is `results/task2c_probe_results.csv`.

| Strategy | P08 | P10 | Official accuracy |
|---|---|---|---:|
| Full | Correct | Correct | 2/2 (100%) |
| Trim | Correct | Correct | 2/2 (100%) |
| Summary | Correct | Correct | 2/2 (100%) |

These six judgments are semantic review, not substring matching. P08 explains that `ERR-RMA-04` means a return reason is missing or invalid, cites the retrieved active KB-007 chunk, and contains no material factual error. The recorded response is a deterministic fallback because prompt preparation exceeded its safe token budget; it is still grounded and answers the probe. P10 preserves 28 November 2026, merges the after-6-pm constraint, checks actual slot availability, and offers the two returned windows for selection. Full also used a fallback for P10; the fallback correctly offers the actual options. The 2/2 result per strategy is limited to this two-probe Session A inventory; it does not demonstrate general recall quality.

No official P08/P10 probe failed in these runs. Therefore there is no genuine official failed probe/source-turn pair to report for any strategy. This is an evaluation-dataset limitation, not a reason to manufacture a failure.

## Supplemental P01/P03 memory diagnostics

P01 and P03 are Session B probes used only as supplemental short-term-memory diagnostics against the Session A snapshots. They are not included in official accuracy and do not test Task 3 persistent memory. Raw inputs, selected context, actual summary, response, status, and call metadata are preserved in `results/task2c_supplemental_memory_diagnostics.json` and `results/task2c_supplemental_runs/`. Each starts from a fresh snapshot clone, uses the same tracker snapshot across strategies, and bypasses DialogueManager routing, tools, retrieval, and persistent memory.

- **Full:** P01 and P03 both failed before an Ollama inference call because the complete history exceeded the safe prompt budget (3,035 and 3,031 tokens against 1,888). These are recorded as `context_budget_failure`, not semantic-recall failures.
- **Trim:** P01 returned the active return pickup's “after 6 pm” constraint for the saved delivery-window question; P03 returned pickup/booking detail instead of SMS. The selected history held only turns 23–25, so the source facts at turns 19 and 20 were evicted. The tracker snapshot still contained unrelated return-workflow state, which is visible in the captured prompt and plausibly contributed to the substitution. This is a verified context mismatch, while the precise model reasoning remains an inference.
- **Summary:** P01 and P03 responses likewise substituted return-pickup facts and included unsupported claims that the pickup was confirmed/being processed. The actual final summary snapshot is nested/wrapper-like generated text and omits both the current `19:00–21:00` preference and SMS. The final summary cursor is 22. We can verify omission by the final snapshot, but the artifacts do not record each intermediate summary version, so the exact update that first lost/distorted each fact cannot be identified.

These supplemental failures are reported individually and do not alter the official 100% P08/P10 scores. They show meaningful memory-context weaknesses for Trim and Summary and prompt-budget failures for Full. No correction is inferred from these diagnostics.

## Trim's first raw-history eviction

An exchange is one complete user/assistant pair; the incoming user message is added separately. The earliest Session A user fact is the preferred-name instruction at turn 2. At incoming turn 6, the last three completed exchanges are turns 3, 4 and 5, so turn 2 is the first fact to leave Trim's selected raw history. The email preference from turn 4 leaves later, at incoming turn 8, when completed exchanges 5, 6 and 7 are selected. Thus the initial turn-8 estimate applies to the email fact, not the earliest user fact. This first window eviction is distinct from an actual incorrect official probe answer: neither official probe depends on the turn-2 or turn-4 fact, and neither failed.

## State, booking, and data integrity

Turn 15 state-follow-up evidence, the Session B turn 33 isolated integration test, and turn 37 clarification evidence are in `results/task2c_state_followup_and_clarification.md`. The turn 33 test replays exact Session B turns 31?33 with fresh state and a mocked KB gateway; its test-only response is not a runtime Qwen answer. It explains the general post-pickup process conditionally. Determining whether the specific Session A pickup occurred requires persistent episode state from Task 3, which is not simulated. Separate real tool-gateway booking and confirmation evidence is in `results/task2c_booking_confirmation_evidence.md`. The replay runs themselves made no booking. The integration tests prove a selected slot is held pending explicit matching confirmation before a write, and reject duplicate booking.

The Task 2(c) run manifest records 25/25 turns for every run, matching reset fingerprints, separate transcripts and restorable snapshots. Summary text/redaction state is retained in its snapshot. The supplemental evaluator validates tracker equality and restores fresh memory and database copies. `logs/calls.jsonl` remains append-only; old Task 2(b) call records were not edited.

## Task 2(a) labelled NLU verification

The current rule-based NLU was independently applied to all 40 labelled cases in `data/nlu_test_cases.json`, including the explicitly specified expected-entity values. Result: **40/40 correct (100%)**, with no intent or expected-entity mismatches. `results/nlu_accuracy.json` reports the same score; the data is the 40 labelled utterances from `data/test_conversation.json`, turns 1–40. This is accuracy on the supplied scripted set, not a claim about unseen traffic.

## Limitations

- The official Session A probe inventory has only P08 and P10 and does not test old-fact recall; supplemental P01/P03 are clearly separated.
- Full's complete history reaches the prompt budget, so its supplemental evaluations cannot reach Qwen.
- Summary quality is weak in the captured final summary and the update-by-update versions were not retained, limiting attribution to an exact summary-update turn.
- Session B turn 33 was not runtime-replayed; an isolated integration test replays exact turns 31?33 and verifies a general conditional answer from KB-003/KB-014. It cannot establish whether the specific Session A pickup occurred; that requires Task 3 persistence. Turn 37 is verified by an isolated test, not a Session B runtime execution.
- Task 2(c) replay and supplementary diagnostics were actual Qwen/Ollama executions; tests validate the booking gate separately using an isolated temporary database.
