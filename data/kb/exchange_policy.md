ShopSphere Product Exchange Policy
Document ID: KB-004 | Version: 1.1 | Effective: 2026-01-01 | Status: active

Exchange options
An exchange replaces an eligible item with another item or variant rather than returning the purchase amount. Exchanges are subject to the same delivery-date eligibility windows as returns: 30 days for most eligible products and 14 days for electronics. The exchange service checks stock in the destination fulfilment region before presenting a replacement option. A replacement is not reserved until the customer confirms the exchange summary and the exchange record is created.

Eligible exchange reasons
Customers may request an exchange for a size or colour change, a wrong variant, or a verified product fault. A fault-related exchange may require diagnostic questions and inspection. If the requested replacement has a higher price, the customer must pay the displayed difference before confirmation. If the replacement is lower priced, the difference is handled through the refund process described in KB-003. The assistant must not invent an available replacement or a price difference; it must read current catalogue and stock results.

Exchange versus refund
A customer may choose an eligible refund instead of an exchange. The assistant should explain both paths neutrally when asked, then ask which option the customer prefers. Once an exchange is created, the same item cannot simultaneously have an open refund request. If a customer changes their mind, the current exchange state must be checked before cancelling or replacing it.

Process and confirmation
The exchange flow requires an order ID, item ID, reason, replacement selection and delivery address confirmation. The dialogue manager must collect missing slots and summarise the chosen replacement, any price difference and the expected dispatch window. The write operation is executed only after an explicit confirmation such as “Yes, place the exchange.” A vague response such as “sounds good” is not sufficient if the summary is not visible in the immediately preceding turn.

Stock and failed exchange
If the replacement is out of stock, offer a waitlist or refund option only when those options are returned by the order service. A failed stock check must not be converted into a successful exchange. Record a failure code and explain the next available action. If stock changes after the quote, revalidate it immediately before the database write.

Customer expectations
An exchange may create a new shipment and a reverse pickup. The customer should receive separate references for the return leg and replacement shipment. Do not promise that the replacement arrives on the same day as the original pickup unless the logistics service explicitly confirms that commitment.
