# Task 2(c): State follow-up and clarification evidence

## Session A turn 15 (official replay evidence)

All three Task 2(c) replays had the same tracker state immediately before turn 15: order `ORD-74021`, item `SKU-AUR-220`, pickup date `2026-11-28`, and no pickup time window. The incoming message was “And for Saturday, please make it after 6 pm.” NLU classified it as `state_followup` and extracted `{start: 18:00, end: null, raw: after 6 pm}`. The tracker merged this with the existing date rather than starting a new task.

The actual scheduler result in each run offered two available slots for 28 November: `PICK-20261128-E1` (18:00–20:00) and `PICK-20261128-E2` (20:00–22:00). Each response presented both choices and asked the customer to choose. No slot was selected and no booking was made at turn 15. The state before/after, exact tool result, response, NLG status and citations are in the `P10` rows of `results/task2c_probe_results.csv` and the original transcripts linked by the Task 2(c) manifest.

## Session B clarification case (separate from the official Session A accuracy set)

The scripted Session B turn 37 is: “Can you book a pickup for the other item as well?” With no order or item identified, the expected safe response is: “Which order and item do you mean? Please provide the order ID or select an item.” The isolated regression `test_session_b_other_item_request_clarifies_without_any_tool_call` verifies this exact message and response, missing `order_id`/`sku`, and zero tool calls. This is test evidence, not a claim that Session B was replayed. Since no tool executes, no unauthorized database write can occur.

Session B turn 33 (“And what happens next after the pickup is completed?”) is labelled `state_followup` in the source cases. No Session B runtime replay was run, so there is no observed assistant response to report for that case.

## Session B turn 33 closure test (isolated integration test, not a runtime replay)

The source dataset's exact preceding Session B messages are:

| Turn | User message | Expected behavior |
|---:|---|---|
| 31 | “For a 2026 delivery, how many days do I have to return electronic accessories?” | Retrieve current KB-002 only and cite it. |
| 32 | “Can you check the latest status of order ORD-74021?” | Call `get_order` and report the database status. |
| 33 | “And what happens next after the pickup is completed?” | Resolve the follow-up; explain receipt, inspection and conditional refund stages using KB-003. |

Probe P11 identifies source turns 23 and 32 and KB-003. The corrected Session A replay did not create a booking at turn 23, so the probe's source turn is not evidence that a pickup was actually confirmed. Turn 32's seed database result is `ORD-DEL`; the seed order has `existing_return: null`. The fresh Session B tracker after turn 32 has no active return and no `order_id` slot. The test does not import Session A tracker state, claim that the earlier pickup occurred, or create a return/booking event. Turn 33 can still be answered as a **general conditional process** from KB evidence. The specific status of a Session A pickup would require persistent cross-session episode/state retrieval, which is outside Task 2(c) and remains for Task 3; it is not simulated here.

When no return workflow is active, the Dialogue Manager now routes a `state_followup` to KB retrieval. An active `return_pickup` workflow continues through the existing slot-filling path. The focused test `test_session_b_turn_33_uses_preceding_context_and_kb_for_general_pickup_followup` replays the exact user messages for turns 31–33 against an isolated tracker and a mocked tool gateway. The gateway returns source text from the KB-003 lifecycle passage and the KB-014 passage supporting receipt at the returns centre. A test-only response generator composes a conditional answer from those retrieved passages; this is not represented as an observed Qwen or Session B runtime response.

Captured test evidence:

- **NLU:** turn 33 is classified `state_followup` (rule `R14_STATE_FOLLOWUP`).
- **Prior context/state:** turn 31 asks the current electronic-accessory return window; turn 32 reports `ORD-74021` as `ORD-DEL`; the tracker has `active_task: null` and no return `order_id` slot.
- **Retrieval/tool evidence:** turn 33 calls `search_kb` for the exact follow-up. The mocked result includes KB-003 version 1.3 (`item received`, `inspection completed`, and `refund initiated`) and KB-014 version 1.0 (inspection update after receipt at the returns centre). KB-003 supplies the lifecycle; the more specific returns-centre wording is supported by KB-014. Sources and candidate/result counts are asserted in the test.
- **Test response:** “After collection, the return is received at the returns centre and inspected. If accepted, the refund is then initiated; this does not confirm that this pickup or a refund has occurred.”
- **Grounding/safety assertions:** the test verifies every stage against the returned evidence and rejects claims that this pickup is complete, that this refund has been initiated, or that a refund reference exists.
- **Test result:** `1 passed` for the focused turn 33 test. The complete suite was rerun after this change; see the final validation report.

## Relevant regression coverage

- `tests/test_dialogue_manager.py`: active state follow-up, turn 33 general KB follow-up, missing item clarification, turn 37 ambiguity, and mismatched confirmation rejection.
- `tests/test_state_tracker.py`: state preservation and explicit-confirmation gates.
- `tests/test_live_tool_integration.py`: real JSON-backed slot lookup and confirmation-gated booking.
