ShopSphere Refund Processing Guide
Document ID: KB-003 | Version: 1.3 | Effective: 2026-02-01 | Status: active

Refund lifecycle
A refund has several distinct stages: return requested, return authorised, item collected, item received, inspection completed, refund initiated and payment provider posting. A customer may see the return marked accepted before the credit is visible in the bank account. Support must communicate the current stage exactly and must not describe an initiated refund as a settled bank credit. Each transition is timestamped in the order service.

Indicative processing estimates
After an accepted inspection, ShopSphere normally initiates a card refund within 2 business days. A UPI refund is normally initiated within 1 business day. A net-banking refund is normally initiated within 3 business days. A ShopSphere wallet refund is normally initiated within 4 business hours. These are initiation estimates, not guarantees for the customer's bank. The bank, card issuer or payment network may require additional time to post the credit. Weekends and bank holidays are not business days for the card and net-banking estimates.

Refund identifiers
A refund event receives a reference such as RFND-2026-10482. The payment service also records the original payment transaction reference and the masked destination. Support may share the refund reference and initiation date, but must never disclose full card numbers, UPI PINs, passwords or authentication codes. If a customer asks for a UPI PIN or OTP to “verify” a refund, explain that ShopSphere will never request it.

Partial refunds
If only one item in a multi-item order is accepted, the refund is calculated for that item and any item-specific discount allocation. Shipping charges are not automatically refunded when a customer returns one item because of change of mind. If the return is caused by a confirmed ShopSphere fulfilment error, the shipping charge may also be credited after review. The final amount is calculated by the refund service; the assistant should read that amount rather than calculate a promise from the list price.

Delayed credits
If the stated initiation estimate has passed, first check the refund event and its status. If no refund event exists, route the case to returns operations. If the event is initiated but the credit is not visible, provide the reference and suggest that the customer check with the payment provider. Escalate to Payments Support when the provider trace is older than five business days for UPI or ten business days for card or net banking. Never create a second refund merely because a customer cannot yet see the first credit.

Status language
Use “initiated” when ShopSphere has sent the refund instruction, “processing” when the provider acknowledges it, and “posted” only when the provider confirms the credit. When provider status is unavailable, disclose that limitation and provide the last verified timestamp.
