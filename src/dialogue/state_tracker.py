"""Session-level dialogue state tracking for RecallBot V05 (Task 2a).

The tracker keeps the current task's slots and confirmation state separate
from the raw conversation history. It does not call an LLM or execute tools.
The dialogue manager is responsible for invoking tools after inspecting this
state. In particular, this module never performs a database write.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_REQUIRED_SLOTS = (
    "customer_id",
    "order_id",
    "sku",
    "reason",
    "pickup_date",
    "pickup_time_window",
    "pickup_address_summary",
    "explicit_confirmation",
)

# Words that do not help identify a product in an order's item list.
_ITEM_HINT_STOP_WORDS = {
    "a", "an", "and", "as", "book", "can", "for", "from", "i",
    "item", "my", "need", "of", "order", "please", "return", "the",
    "to", "want",
}


def _load_required_slots(workflow_path: Path | None = None) -> tuple[str, ...]:
    """Read the authoritative slot list from the versioned workflow file.

    A small fallback keeps the tracker independently testable if the workflow
    file is not present. In the project, the JSON file is expected to exist.
    """
    if workflow_path is None:
        project_root = Path(__file__).resolve().parents[2]
        workflow_path = project_root / "skills" / "return_pickup_workflow_v1.json"
    if workflow_path.exists():
        payload = json.loads(workflow_path.read_text(encoding="utf-8"))
        slots = payload.get("required_slots")
        if not isinstance(slots, list) or not slots or not all(isinstance(x, str) for x in slots):
            raise ValueError(f"Invalid required_slots in workflow: {workflow_path}")
        return tuple(slots)
    return DEFAULT_REQUIRED_SLOTS


def _normalise_words(value: str) -> set[str]:
    return {
        word.casefold()
        for word in re.findall(r"[a-zA-Z0-9]+", value)
        if len(word) > 2 and word.casefold() not in _ITEM_HINT_STOP_WORDS
    }


class DialogueStateTracker:
    """Maintain a structured, session-local state for dialogue orchestration.

    Typical flow:
        tracker = DialogueStateTracker(session_id="A")
        tracker.update_from_nlu(text, analyze_utterance(text), turn=12)
        tracker.apply_order_result(verified_order, item_hint=text)
        readiness = tracker.request_confirmation(action_payload)
        # After the user explicitly confirms:
        tracker.update_from_nlu(confirm_text, confirm_nlu, turn=23)
        action = tracker.consume_confirmed_action()  # exactly once

    The caller, not this class, executes the returned action through a tool.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        required_slots: Sequence[str] | None = None,
        workflow_path: Path | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

        self.required_slots = tuple(required_slots or _load_required_slots(workflow_path))
        if "explicit_confirmation" not in self.required_slots:
            raise ValueError("Workflow required_slots must include explicit_confirmation")

        slots: dict[str, Any] = {name: None for name in self.required_slots}
        slots["explicit_confirmation"] = False
        self._state: dict[str, Any] = {
            "session_id": session_id,
            "user_id": user_id,
            "turn": 0,
            "last_user_text": None,
            "current_intent": "unknown",
            "current_entities": {},
            "active_task": None,
            "task_status": "idle",
            "slots": slots,
            "missing_slots": list(self.required_slots),
            "available_order_items": [],
            "clarification_required": False,
            "clarification_prompt": None,
            "pending_confirmation": False,
            "confirmation_status": "not_requested",
            "confirmation_mismatch": False,
            "pending_action": None,
            "database_write_authorized": False,
            "last_tool_result": None,
        }
        self._refresh_missing_slots()

    @property
    def state(self) -> dict[str, Any]:
        """Return a defensive copy so callers cannot mutate tracker state."""
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a serialized state for an isolated evaluation replay."""
        if not isinstance(snapshot, Mapping):
            raise TypeError("tracker snapshot must be a mapping")
        slots = snapshot.get("slots")
        if not isinstance(slots, Mapping) or set(self.required_slots) - set(slots):
            raise ValueError("tracker snapshot slots do not match the configured workflow")
        if snapshot.get("session_id") != self._state["session_id"]:
            raise ValueError("tracker snapshot session does not match this tracker")
        if snapshot.get("user_id") != self._state["user_id"]:
            raise ValueError("tracker snapshot user does not match this tracker")
        restored = deepcopy(dict(snapshot))
        self._state = restored
        self._refresh_missing_slots()

    def missing_slots(self, *, include_confirmation: bool = True) -> list[str]:
        """Return required slots that have not yet been collected."""
        missing: list[str] = []
        for name in self.required_slots:
            if name == "explicit_confirmation" and not include_confirmation:
                continue
            value = self._state["slots"].get(name)
            if value is None or value is False or value == "":
                missing.append(name)
        return missing

    def _refresh_missing_slots(self) -> None:
        self._state["missing_slots"] = self.missing_slots()

    def set_slot(self, name: str, value: Any) -> None:
        """Set a workflow slot, except confirmation which needs a user turn."""
        if name not in self.required_slots:
            raise KeyError(f"Unknown workflow slot: {name}")
        if name == "explicit_confirmation":
            raise ValueError("Use an explicit confirm_action turn to set confirmation")
        if value is None or value == "":
            raise ValueError(f"Slot {name!r} cannot be set to an empty value")
        self._state["slots"][name] = deepcopy(value)
        self._refresh_missing_slots()

    def update_from_nlu(
        self,
        utterance: str,
        nlu_result: Mapping[str, Any],
        *,
        turn: int,
    ) -> dict[str, Any]:
        """Merge one NLU result into the current task state.

        State-dependent follow-ups are merged with existing slots. For example,
        "And for Saturday, after 6 pm?" updates the pickup window without
        discarding the pickup date collected on the previous turn.
        """
        if not isinstance(utterance, str):
            raise TypeError("utterance must be a string")
        if not isinstance(turn, int) or turn < 1:
            raise ValueError("turn must be a positive integer")
        if not isinstance(nlu_result, Mapping):
            raise TypeError("nlu_result must be a mapping")

        intent = nlu_result.get("intent", "unknown")
        entities = nlu_result.get("entities", {})
        if not isinstance(intent, str) or not isinstance(entities, Mapping):
            raise ValueError("NLU result must contain a string intent and entity mapping")

        self._state["turn"] = turn
        self._state["last_user_text"] = utterance
        self._state["current_intent"] = intent
        self._state["current_entities"] = deepcopy(dict(entities))
        self._state["clarification_required"] = intent == "clarification_required"
        self._state["clarification_prompt"] = (
            "Which order and item do you mean? Please provide the order ID or select an item."
            if self._state["clarification_required"] else None
        )

        if intent == "start_return":
            if self._state["active_task"] not in (None, "return_pickup"):
                raise RuntimeError("Cannot start a return pickup while another task is active")
            self._state["active_task"] = "return_pickup"
            if self._state["task_status"] in {"idle", "cancelled", "completed"}:
                self._state["task_status"] = "collecting_slots"

        # Only slot-collection turns may bind extracted entities to the active
        # return. Mentions in historical decisions, complaints, confirmations,
        # and unrelated requests remain recorded in current_entities only.
        return_slot_intents = {
            "start_return", "provide_return_reason", "provide_pickup_date",
            "provide_pickup_window", "state_followup", "unknown",
        }
        if self._state["active_task"] == "return_pickup" and intent in return_slot_intents:
            order_ids = entities.get("order_ids") or []
            if order_ids:
                self.set_slot("order_id", order_ids[0])

            skus = entities.get("skus") or []
            if skus:
                self.set_slot("sku", skus[0])
                self._state["clarification_required"] = False
                self._state["clarification_prompt"] = None

            reason = entities.get("return_reason")
            if reason:
                self.set_slot("reason", reason)

            date_value = entities.get("date")
            if date_value:
                self.set_slot("pickup_date", date_value)

            time_window = entities.get("time_window")
            if time_window:
                self.set_slot("pickup_time_window", dict(time_window))

        if intent == "confirm_action":
            # A confirmation is meaningful only after the tracker has placed a
            # specific proposed action in the pending-confirmation state.
            if self._state["pending_confirmation"] and self._state["pending_action"]:
                arguments = self._state["pending_action"].get("arguments", {})
                claimed_orders = entities.get("order_ids") or []
                claimed_skus = entities.get("skus") or []
                expected_order = str(arguments.get("order_id") or "").casefold()
                expected_sku = str(arguments.get("sku") or "").casefold()
                mismatch = (
                    any(str(value).casefold() != expected_order for value in claimed_orders)
                    or any(str(value).casefold() != expected_sku for value in claimed_skus)
                )
                self._state["confirmation_mismatch"] = mismatch
                if not mismatch:
                    self._state["slots"]["explicit_confirmation"] = True
                    self._state["pending_confirmation"] = False
                    self._state["confirmation_status"] = "confirmed"
                    self._state["pending_action"]["status"] = "confirmed"
                    self._state["task_status"] = "confirmed"
            # Otherwise leave confirmation false; never infer permission.

        elif intent == "cancel_action" and self._state["pending_confirmation"]:
            self._state["pending_confirmation"] = False
            self._state["confirmation_mismatch"] = False
            self._state["confirmation_status"] = "cancelled"
            if self._state["pending_action"]:
                self._state["pending_action"]["status"] = "cancelled"
            self._state["task_status"] = "cancelled"
            self._state["slots"]["explicit_confirmation"] = False

        self._refresh_missing_slots()
        return self.snapshot()

    def apply_order_result(
        self,
        order: Mapping[str, Any],
        *,
        item_hint: str | None = None,
    ) -> dict[str, Any]:
        """Apply a successful, already-authorised get_order result.

        Ownership verification belongs in the order tool. This method only
        maps its returned customer/address/item details into dialogue slots.
        An item is auto-selected only when the utterance clearly matches one
        unique order item; otherwise the tracker asks for clarification.
        """
        if not isinstance(order, Mapping):
            raise TypeError("order result must be a mapping")
        order_id = order.get("order_id")
        if not order_id:
            raise ValueError("order result is missing order_id")
        requested_order_id = self._state["slots"].get("order_id")
        if requested_order_id and str(requested_order_id).casefold() != str(order_id).casefold():
            raise ValueError("get_order result does not match the active order_id")
        if self._state["active_task"] != "return_pickup":
            raise RuntimeError("Order details can only be applied to an active return_pickup task")

        self.set_slot("order_id", str(order_id).upper())
        if order.get("customer_id"):
            self.set_slot("customer_id", str(order["customer_id"]))
            self._state["user_id"] = str(order["customer_id"])
        if order.get("address_summary"):
            self.set_slot("pickup_address_summary", str(order["address_summary"]))

        items = order.get("items") or []
        self._state["available_order_items"] = [
            {"sku": item.get("sku"), "name": item.get("name")}
            for item in items if isinstance(item, Mapping)
        ]

        hint = item_hint if item_hint is not None else (self._state.get("last_user_text") or "")
        hint_words = _normalise_words(hint)
        scored: list[tuple[int, Mapping[str, Any]]] = []
        for item in items:
            if not isinstance(item, Mapping) or not item.get("sku"):
                continue
            name_words = _normalise_words(str(item.get("name", "")))
            score = len(hint_words & name_words)
            if score:
                scored.append((score, item))

        selected: Mapping[str, Any] | None = None
        if scored:
            best_score = max(score for score, _ in scored)
            best_items = [item for score, item in scored if score == best_score]
            if len(best_items) == 1:
                selected = best_items[0]
        elif len(items) == 1 and isinstance(items[0], Mapping):
            selected = items[0]

        if selected:
            self.set_slot("sku", str(selected["sku"]).upper())
            self._state["clarification_required"] = False
            self._state["clarification_prompt"] = None
        else:
            self._state["slots"]["sku"] = None
            self._state["clarification_required"] = True
            self._state["clarification_prompt"] = (
                "This order contains more than one item. Which item would you like to return?"
            )

        self._state["last_tool_result"] = {
            "tool": "get_order",
            "status": "success",
            "order_id": str(order_id).upper(),
            "item_count": len(items),
        }
        self._refresh_missing_slots()
        return self.snapshot()

    def request_confirmation(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Place one specific tool action in a pending-confirmation state.

        The action is only recorded. No tool or database write is executed.
        The dialogue manager should display a human-readable summary and ask
        the user before accepting a confirmation turn.
        """
        if self._state["active_task"] != "return_pickup":
            raise RuntimeError("No active return_pickup task to confirm")
        missing = self.missing_slots(include_confirmation=False)
        if missing:
            return {"ready": False, "missing_slots": missing, "state": self.snapshot()}
        if not isinstance(action, Mapping) or not action.get("tool"):
            raise ValueError("action must identify the tool to be called")
        if self._state["pending_confirmation"] or self._state["confirmation_status"] == "confirmed":
            raise RuntimeError("An action is already awaiting confirmation or execution")

        self._state["pending_action"] = {
            "tool": str(action["tool"]),
            "arguments": deepcopy(action.get("arguments", {})),
            "status": "awaiting_confirmation",
        }
        self._state["pending_confirmation"] = True
        self._state["confirmation_mismatch"] = False
        self._state["confirmation_status"] = "pending"
        self._state["task_status"] = "awaiting_confirmation"
        self._state["slots"]["explicit_confirmation"] = False
        self._refresh_missing_slots()
        return {"ready": True, "missing_slots": [], "state": self.snapshot()}

    def consume_confirmed_action(self) -> dict[str, Any] | None:
        """Return a confirmed action once, for the dialogue manager to execute.

        Repeated calls return None. This prevents a repeated confirmation turn
        from causing the same database action to be dispatched twice by code
        that follows this interface.
        """
        pending = self._state.get("pending_action")
        if not pending or self._state["confirmation_status"] != "confirmed":
            return None
        if pending.get("status") != "confirmed":
            return None
        pending["status"] = "execution_started"
        self._state["database_write_authorized"] = True
        self._state["task_status"] = "executing"
        return {
            "tool": pending["tool"],
            "arguments": deepcopy(pending.get("arguments", {})),
        }

    def record_action_result(self, tool_result: Mapping[str, Any]) -> None:
        """Record a real tool result after the dialogue manager executes it."""
        pending = self._state.get("pending_action")
        if not pending or pending.get("status") != "execution_started":
            raise RuntimeError("No confirmed action is currently executing")
        if not isinstance(tool_result, Mapping):
            raise TypeError("tool_result must be a mapping")
        self._state["last_tool_result"] = deepcopy(dict(tool_result))
        pending["status"] = "completed"
        self._state["database_write_authorized"] = False
        self._state["task_status"] = "completed"
        self._refresh_missing_slots()

    def record_action_failure(self, error_type: str) -> None:
        """Record a failed real tool call without claiming success."""
        pending = self._state.get("pending_action")
        if not pending or pending.get("status") != "execution_started":
            raise RuntimeError("No confirmed action is currently executing")
        pending["status"] = "failed"
        self._state["database_write_authorized"] = False
        self._state["task_status"] = "action_failed"
        self._state["last_tool_result"] = {
            "tool": pending["tool"],
            "status": "failed",
            "error_type": str(error_type),
        }

    def clear_active_task(self) -> None:
        """Clear the completed/cancelled workflow without touching other memory."""
        if self._state["task_status"] not in {"completed", "cancelled", "action_failed", "idle"}:
            raise RuntimeError("Cannot clear an active or unconfirmed task")
        for name in self.required_slots:
            self._state["slots"][name] = False if name == "explicit_confirmation" else None
        self._state.update({
            "active_task": None,
            "task_status": "idle",
            "pending_confirmation": False,
            "confirmation_status": "not_requested",
            "pending_action": None,
            "confirmation_mismatch": False,
            "database_write_authorized": False,
            "clarification_required": False,
            "clarification_prompt": None,
            "available_order_items": [],
        })
        self._refresh_missing_slots()
