# Task 2(c): Booking confirmation evidence

## Historical Task 2(c) Session A runs

The three recorded strategy replays did not create a booking. Turn 15 returned two matching pickup windows and asked the customer to select one; turn 22 repeated the options. The later confirmation did not authorize a write because no specific pending slot action had been created. Each final runtime database has an empty `bookings` list. These are the historical replay outcomes; they are not a successful-booking demonstration.

## Separate real tool integration test

`tests/test_live_tool_integration.py::test_user_can_select_displayed_24_hour_slot_label_verbatim` and `::test_real_database_write_runs_only_after_dialogue_manager_confirmation` use the real JSON-backed tool gateway with a temporary runtime database:

1. With multiple available windows, neither an availability response nor the slot-choice prompt writes a booking.
2. Selecting `18:00–20:00` creates a pending action containing `ORD-74021`, `SKU-AUR-220`, reason `defective_left_earbud`, date `2026-11-28`, window `18:00–20:00`, masked address summary, and slot ID `PICK-20261128-E1`. The confirmation summary displays the same identifying values.
3. Before explicit confirmation, the temporary database still has no booking.
4. A separate “Yes, I confirm the pickup booking” turn passes the tracker gate and invokes the real tool. The persisted JSON has one booking with `confirmation_received: true` and the selected slot ID.

Related safeguards are covered by:

- `tests/test_dialogue_manager.py::test_confirmation_details_must_match_the_pending_booking`: a mismatched order/SKU confirmation is refused, pending action remains, and `book_slot` is not called; a subsequent valid confirmation is accepted.
- `tests/test_tools.py::test_booking_rejects_missing_explicit_confirmation_without_write`: missing explicit confirmation is rejected without a write.
- `tests/test_tools.py::test_confirmed_booking_writes_runtime_json_and_prevents_double_booking`: a confirmed booking persists and a duplicate booking is rejected.
- `tests/test_state_tracker.py::test_database_action_is_gated_by_explicit_confirmation_and_consumed_once`: authorization is explicit and single-use.

The integration test's temporary database is isolated from the historical replay databases and is not evidence that a Task 2(c) Session A replay booked a pickup.
