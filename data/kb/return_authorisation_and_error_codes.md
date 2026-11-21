ShopSphere Return Authorisation and Error Codes
Document ID: KB-007 | Version: 1.2 | Effective: 2026-01-01 | Status: active

Return identifiers
A return request receives an RMA identifier after the order, item, return window and reason pass validation. RMA-OPEN means the request is created but pickup is not yet booked. RMA-BOOK means a pickup slot is confirmed. RMA-COL means the carrier collected the item. RMA-REC means the returns centre received the parcel. RMA-INS means inspection is in progress. RMA-ACC means the return was accepted. RMA-REJ means the return was rejected after inspection. RMA-RFD means the refund instruction was initiated. These states are sequential but an item can pause between states while an external service responds.

Error codes
ERR-RMA-01 means the order identifier cannot be found. Ask the customer to verify the order ID; do not guess a similar order. ERR-RMA-02 means the requested SKU is not part of the order. Read the order items and ask the customer to select one of the listed items. ERR-RMA-03 means the item is outside the applicable return window. Explain the policy version and delivery date used. ERR-RMA-04 means a return reason is missing or invalid. Ask a concise question and do not create an RMA until a supported reason is selected. ERR-RMA-05 means the pickup slot is unavailable. Offer only slots returned by the scheduling service. ERR-RMA-06 means a confirmation token or explicit customer confirmation is missing. Do not retry the write until confirmation has been obtained.

Authorisation checks
The return service validates order ownership, delivery status, item quantity, return window and any existing open return. If an item already has an open RMA, the assistant should retrieve that RMA rather than creating a duplicate. A write timeout does not prove that the write failed; read the order return history before retrying. This prevents duplicate pickups.

Failure handling
The assistant should explain errors in plain language and state the next action. Internal stack traces, authentication tokens and payment credentials must never be shown. A support case can be created for unresolved exceptions, but the assistant must not represent a pending case as a completed return. For ERR-RMA-06, the user must see the item, reason, pickup date and time window before the final confirmation request.

Audit fields
Each RMA event records order ID, SKU, customer ID, reason code, policy version, request timestamp, pickup slot, confirmation source and actor. These fields make it possible to reconstruct why a request was accepted or declined. The assistant should include the RMA ID in its response when one has actually been created.
