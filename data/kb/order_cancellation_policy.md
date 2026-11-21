ShopSphere Order Cancellation Policy
Document ID: KB-011 | Version: 1.1 | Effective: 2026-01-01 | Status: active

When an order can be cancelled
An order can normally be cancelled while it is in ORD-NEW or ORD-PROC, provided the fulfilment centre has not begun a restricted packing or dispatch step. Once the order is handed to the carrier and reaches ORD-SHIP, the standard cancellation action is unavailable; the customer may need to use the return process after delivery. The order service is the authority on whether cancellation is still possible. The assistant must not infer cancellation eligibility from an estimated delivery date.

Cancellation request flow
Collect the order ID and reason. Read the order state and check for existing cancellation requests. If cancellation is available, show the order items and the cancellation consequence, including any payment reversal estimate returned by the payment service. Ask the customer to confirm the cancellation. Only after an explicit confirmation should the assistant invoke the cancellation write. A request such as “Can I cancel?” is a question, not authorisation to cancel.

Partial cancellation
Some orders contain several items or shipments. Partial cancellation is allowed only when the order service exposes an item-level cancellation action. If one item has already shipped, the assistant must not claim the whole order is cancelled. It should explain which lines can still be cancelled and ask the customer how to proceed. Discounts and shipping charges may be recalculated when an item is removed.

Payment reversal
If payment was authorised but not captured, cancellation may result in an authorisation reversal rather than a refund. If payment was captured, the payment service initiates a refund under the applicable method timeline in KB-003. The two events should not be described as identical. A pending provider response should be communicated as pending.

Failure and recovery
If the cancellation service returns a timeout, retrieve the order again before retrying. A timeout may occur after the write has succeeded. Duplicate cancellation attempts can create confusing audit records, so read-after-write verification is required. If the order is no longer cancellable, explain the next legitimate option instead of attempting a bypass.
