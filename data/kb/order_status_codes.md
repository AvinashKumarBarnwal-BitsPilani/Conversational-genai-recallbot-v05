ShopSphere Order and Shipment Status Codes
Document ID: KB-006 | Version: 1.4 | Effective: 2026-01-01 | Status: active

How to read a status
An order can have an order-level status and a shipment-level status. The order-level status describes the commercial transaction; the shipment-level status describes carrier movement. A customer-facing response should use the plain-language description and include the code only when it helps the customer or support agent. Do not infer that a parcel is physically moving from an order-level status alone.

Order status codes
ORD-NEW means the order was accepted but payment or fraud checks may still be pending. ORD-PROC means payment is confirmed and the fulfilment centre is preparing the items. ORD-HOLD means fulfilment has been paused for an address, payment or inventory check. ORD-PICK means items have been picked and are undergoing packing verification. ORD-SHIP means the parcel has been handed to the carrier. ORD-DEL means the carrier has recorded delivery. ORD-CANC means the order was cancelled before dispatch. ORD-RET means a return record exists; it does not necessarily mean the refund is complete.

Shipment scan codes
SHP-LBL means a shipping label was created but the carrier has not yet scanned the parcel. SHP-PICK means the carrier collected the parcel. SHP-TRANS means the parcel is moving between carrier facilities. SHP-OUT means the parcel is out for delivery. SHP-ATT means a delivery attempt was recorded. SHP-DEP means the parcel is held at a local depot. SHP-DLV means the carrier recorded successful delivery. SHP-EXC means an exception needs investigation.

Status conflicts
If the order service says ORD-SHIP while the carrier says SHP-ATT, explain that the order was dispatched and a delivery attempt was recorded. If the order service says ORD-DEL but the customer disputes receipt, open a proof-of-delivery investigation rather than changing the status manually. When two systems disagree, report the timestamps and route the conflict to operations.

What the assistant may do
The assistant may read status and explain the code. It may not manually change a carrier scan or mark an order delivered. Status updates must come from the authorised order or carrier integration. A stale cached status should be refreshed before making a promise about delivery timing.
