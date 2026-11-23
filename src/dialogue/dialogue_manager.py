"""Dialogue orchestration for RecallBot V05 (Task 2a).

The manager coordinates the existing rule-based NLU and DialogueStateTracker.
Backend capabilities are injected through a small tool-gateway interface so
that orchestration can be unit-tested without making real database writes.

The production tool gateway must:
- enforce customer ownership in get_order();
- return retrieved source metadata from search_kb();
- return only currently available slots from check_pickup_slots();
- revalidate slot availability and persist exactly once in book_slot().

Natural-language generation is intentionally an injectable dependency. Until
an Ollama response generator is connected, deterministic templates are used
for predictable orchestration tests; these templates are not the final Qwen
NLG implementation required by Task 2(a).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import re
import time
from typing import Any, Protocol

from .nlu import analyze_utterance
from .conversation_history import ConversationHistory
from .rolling_summary import RollingSummaryMemory
from .state_tracker import DialogueStateTracker
from ..logger import log_call, timer


class ToolGateway(Protocol):
    """Backend contract consumed by the dialogue manager.

    Implementations are supplied separately from orchestration. Tool methods
    should log each real tool invocation through src.logger.log_call.
    """

    def get_order(self, order_id: str, customer_id: str | None = None) -> Mapping[str, Any]: ...

    def search_kb(self, query: str, top_k: int = 3) -> Sequence[Mapping[str, Any]]: ...

    def check_pickup_slots(
        self,
        pickup_date: str,
        requested_window: Mapping[str, Any],
        address_summary: str,
    ) -> Sequence[Mapping[str, Any]]: ...

    def book_slot(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


ResponseGenerator = Callable[[str, Sequence[Mapping[str, Any]]], str]


def _default_token_counter(text: str) -> int:
    """Count locally, falling back safely without allowing HF network access."""
    try:
        from ..retrieval.tokenizer import count_tokens, load_tokenizer
        return count_tokens(text, tokenizer=load_tokenizer(local_files_only=True))
    except (OSError, RuntimeError, ImportError, ValueError):
        return (len(text) + 2) // 3 if text else 0


_SLOT_PROMPTS = {
    "customer_id": "I need to verify the customer account before proceeding. Please confirm the account details.",
    "order_id": "Please share the order ID for the item you want to return.",
    "sku": "Which item from this order would you like to return?",
    "reason": "Could you tell me why you want to return this item?",
    "pickup_date": "Which date would you prefer for the pickup?",
    "pickup_time_window": "What pickup time window would work for you?",
    "pickup_address_summary": "Which pickup address should we use?",
}


class DialogueManager:
    """Route each turn using NLU output and the current structured dialogue state."""

    def __init__(
        self,
        *,
        tracker: DialogueStateTracker,
        tools: ToolGateway,
        user_id: str | None = None,
        config: str = "full",
        memory_strategy: str | None = None,
        response_generator: ResponseGenerator | None = None,
        nlg: Any | None = None,
        token_counter: Callable[[str], int] = _default_token_counter,
        logger: Callable[..., Any] | None = log_call,
        top_k: int = 3,
    ) -> None:
        self.tracker = tracker
        self.tools = tools
        self.user_id = user_id or tracker.state.get("user_id")
        selected_strategy = memory_strategy or (config if config in {"full", "trim", "summary"} else "full")
        if selected_strategy not in {"full", "trim", "summary"}:
            raise ValueError("memory_strategy must be 'full', 'trim' or 'summary'")
        self.memory_strategy = selected_strategy
        self.config = selected_strategy
        self.conversation_history = ConversationHistory()
        self.rolling_summary_memory = (
            RollingSummaryMemory(self.conversation_history)
            if selected_strategy == "summary" else None
        )
        self.response_generator = response_generator
        # QwenNLG is optional for unit tests. The interactive demo enables it
        # by default; deterministic drafts remain the safe fallback.
        self.nlg = nlg
        self.token_counter = token_counter
        self.logger = logger
        self.top_k = top_k
        self._workflow = self._load_workflow()

    @staticmethod
    def _load_workflow() -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "skills" / "return_pickup_workflow_v1.json"
        if not path.exists():
            return {"skill": "return_pickup_workflow", "version": "unavailable", "steps": []}
        import json
        return json.loads(path.read_text(encoding="utf-8"))

    def handle_turn(self, text: str, *, turn: int) -> dict[str, Any]:
        """Run a turn and append its completed user/assistant exchange once."""
        history = (
            self.conversation_history.get_history(self.memory_strategy)
            if self.memory_strategy in {"full", "trim"} else []
        )
        self._active_conversation_history = history
        self._active_conversation_summary = ""
        try:
            result = self._handle_turn_internal(text, turn=turn)
        except Exception:
            # Incomplete turns are not committed to conversation history.
            raise
        finally:
            self._active_conversation_history = []
            self._active_conversation_summary = ""
        self.conversation_history.record_exchange(
            {"role": "user", "content": text},
            {"role": "assistant", "content": str(result["reply"])},
        )
        return result

    def _handle_turn_internal(self, text: str, *, turn: int) -> dict[str, Any]:
        """Classify a user turn, update state, route work and return a reply.

        The method never fabricates tool results. A missing or failed backend
        capability produces an explicit fallback response.
        """
        # Pass the current dialogue coordinates to real tools so every tool
        # and embedding call is logged against the correct user/session/turn.
        set_tool_context = getattr(self.tools, "set_call_context", None)
        call_context = {
            "user": str(self.user_id or "UNKNOWN"),
            "session": str(self.tracker.state["session_id"]),
            "turn": turn,
            "config": self.config,
        }
        if callable(set_tool_context):
            set_tool_context(**call_context)
        set_nlg_context = getattr(self.nlg, "set_call_context", None)
        if callable(set_nlg_context):
            set_nlg_context(**call_context)

        nlu_start = timer()
        print("[RecallBot] NLU analysis started")
        nlu = analyze_utterance(text)
        print(f"[RecallBot] NLU analysis completed: {(timer()-nlu_start)*1000:.0f} ms")
        self._log_nlu(text, nlu, turn=turn, start_time=nlu_start)
        print(f"[RecallBot] NLU logging completed: {(timer()-nlu_start)*1000:.0f} ms")

        if self.rolling_summary_memory is not None:
            if nlu.get("intent") == "forget_request":
                forgotten_delivery_note = (nlu.get("entities") or {}).get("delivery_note")
                if isinstance(forgotten_delivery_note, str) and forgotten_delivery_note.strip():
                    self.rolling_summary_memory.add_redaction(forgotten_delivery_note)
            self._prepare_summary_context()

        tracker_started = timer()
        try:
            state = self.tracker.update_from_nlu(text, nlu, turn=turn)
            print(f"[RecallBot] State tracker update completed: {(timer()-tracker_started)*1000:.0f} ms")
        except (RuntimeError, TypeError, ValueError) as exc:
            return self._result(
                f"I couldn't safely update the current task state ({type(exc).__name__}). "
                "Please clarify what you'd like to do.",
                nlu=nlu,
                tool_calls=[],
                needs_user_input=True,
            )

        intent = nlu["intent"]
        calls: list[dict[str, Any]] = []

        # Confirmation and cancellation are handled before any new task routing.
        if intent == "confirm_action":
            return self._handle_confirmation(nlu, calls)
        if intent == "cancel_action":
            return self._result(
                "The pending pickup action has been cancelled. No booking was made.",
                nlu=nlu, tool_calls=calls, needs_user_input=False,
            )

        if intent == "clarification_required":
            prompt = self.tracker.state.get("clarification_prompt") or (
                "Could you clarify which order and item you mean?"
            )
            return self._result(prompt, nlu=nlu, tool_calls=calls, needs_user_input=True)

        if intent == "greeting":
            return self._result(
                "Hello! I'm RecallBot. I can help with ShopSphere orders, delivery, returns and refunds.",
                nlu=nlu, tool_calls=calls,
            )
        if intent in {"small_talk", "closing"}:
            reply = "You're welcome. Is there anything else I can help you with?" if intent == "small_talk" else "Goodbye! Thank you for contacting ShopSphere."
            return self._result(reply, nlu=nlu, tool_calls=calls)
        if intent == "context_budget_query":
            return self._result(
                "This RecallBot variant uses a configured 2,048-token input context budget. "
                "The context manager will select the information included in each prompt.",
                nlu=nlu, tool_calls=calls,
            )
        if intent == "forgotten_memory_query":
            return self._result(
                "I should not recover or repeat information that was explicitly forgotten.",
                nlu=nlu, tool_calls=calls,
            )
        if intent == "forget_request":
            return self._result(
                "I understand the forget request. Persistent-memory deletion is not connected yet, "
                "so I won't claim that the information has been deleted.",
                nlu=nlu, tool_calls=calls, needs_user_input=False,
            )
        if intent in {"semantic_memory_query", "episodic_memory_query"}:
            return self._result(
                "Persistent semantic and episodic memory retrieval is part of the next project task "
                "and is not connected to this dialogue pipeline yet.",
                nlu=nlu, tool_calls=calls,
            )

        if intent == "start_return":
            return self._start_return(text, nlu, calls)

        if intent == "order_status_query":
            return self._order_status(nlu, calls)

        if intent == "knowledge_base_query":
            return self._knowledge_base_answer(text, nlu, calls)

        # An order ID may arrive in a later turn after the bot asks for it.
        # Likewise, a user may answer an item-clarification question in plain
        # language (e.g. "the phone case") without supplying a SKU. Resolve
        # those cases against the already-verified order rather than guessing.
        state = self.tracker.state
        return_routing_intents = {
            "provide_return_reason", "provide_pickup_date", "provide_pickup_window",
            "state_followup", "unknown",
        }
        if state.get("active_task") == "return_pickup" and intent in return_routing_intents:
            slots = state.get("slots", {})
            order_id = slots.get("order_id")
            last_tool = state.get("last_tool_result") or {}
            order_already_loaded = (
                last_tool.get("tool") == "get_order"
                and last_tool.get("status") == "success"
                and last_tool.get("order_id") == order_id
            )
            needs_order_read = bool(order_id) and not order_already_loaded
            needs_item_resolution = (
                bool(order_id)
                and order_already_loaded
                and not slots.get("sku")
                and bool(state.get("available_order_items"))
                and intent == "unknown"
            )
            if intent != "clarification_required" and (needs_order_read or needs_item_resolution):
                return self._load_order_for_active_return(text, nlu, calls, order_id)

        # If a pickup workflow is active, slot-bearing turns update that same
        # workflow. A short follow-up is not treated as a brand-new request.
        state = self.tracker.state
        if state.get("active_task") == "return_pickup" and intent in {
            "provide_return_reason", "provide_pickup_date", "provide_pickup_window",
            "state_followup", "request_task_summary", "unknown",
        }:
            if intent == "request_task_summary":
                # A summary request is not itself confirmation. Revalidate the
                # slots and availability; ambiguous windows still require a choice.
                return self._advance_return_workflow(nlu, calls)
            return self._advance_return_workflow(nlu, calls)

        # A state-follow-up can also ask about the general process after a
        # referenced workflow stage (for example, what happens after pickup).
        # If this session has no active return workflow, answer from retrieved
        # KB evidence; do not manufacture Session A state in a new session.
        if intent == "state_followup":
            return self._knowledge_base_answer(text, nlu, calls)

        if intent in {"provide_user_fact", "update_user_fact"}:
            return self._result(
                "I understood that as a profile preference. Persistent fact storage and updates "
                "will be connected under the long-term memory task.",
                nlu=nlu, tool_calls=calls,
            )

        return self._result(
            "I can help with an order, return, delivery or policy question. "
            "Could you tell me a little more about what you need?",
            nlu=nlu, tool_calls=calls, needs_user_input=True,
        )

    def _log_nlu(self, text: str, nlu: Mapping[str, Any], *, turn: int, start_time: float) -> None:
        if self.logger is None:
            return
        try:
            token_count = int(self.token_counter(text))
        except Exception:
            # Logging must not prevent the conversational pipeline from running.
            token_count = 0
        self.logger(
            config=self.config,
            user=str(self.user_id or "UNKNOWN"),
            session=str(self.tracker.state["session_id"]),
            turn=turn,
            call_type="nlu",
            model="rule-based",
            input_tokens=token_count,
            output_tokens=0,
            retrieved_tokens=0,
            start_time=start_time,
            extra={
                "purpose": "intent_and_entity_extraction",
                "intent": nlu.get("intent", "unknown"),
                "rule_id": nlu.get("rule_id", "R99_NO_MATCH"),
            },
        )

    def _prepare_summary_context(self) -> None:
        """Update aged transcript context before the customer-facing response."""
        memory = self.rolling_summary_memory
        if memory is None:
            return
        summary_generator = getattr(self.nlg, "generate_summary", None)
        if callable(summary_generator):
            started = timer()
            print("[RecallBot] Rolling summary update started")
            try:
                updated = memory.update_summary(summary_generator)
                print(
                    "[RecallBot] Rolling summary update "
                    f"{'completed' if updated else 'not needed'}: {(timer()-started)*1000:.0f} ms"
                )
            except Exception as exc:
                # RollingSummaryMemory leaves both summary and cursor unchanged
                # when the generation callback fails; its context retains every
                # exchange that has not yet been summarized.
                print(
                    "[RecallBot] Rolling summary failed; preserving unsummarized "
                    f"history ({type(exc).__name__}): {(timer()-started)*1000:.0f} ms"
                )
        context = memory.get_context()
        self._active_conversation_summary = context["summary"]
        self._active_conversation_history = context["messages"]

    def _load_order_for_active_return(
        self,
        text: str,
        nlu: Mapping[str, Any],
        calls: list[dict[str, Any]],
        order_id: str,
    ) -> dict[str, Any]:
        """Read the active order and resolve a uniquely identified item."""
        try:
            order = self.tools.get_order(order_id, self.user_id)
            calls.append({"tool": "get_order", "status": "success", "order_id": order_id})
            self.tracker.apply_order_result(order, item_hint=text)
        except Exception as exc:
            calls.append({"tool": "get_order", "status": "failed", "error_type": type(exc).__name__})
            return self._result(
                "I couldn't verify the order details. Please check the order ID or contact ShopSphere support.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        return self._advance_return_workflow(nlu, calls)

    def _start_return(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.tracker.state
        order_id = (nlu.get("entities") or {}).get("order_ids", [None])[0]
        if not order_id:
            return self._result(_SLOT_PROMPTS["order_id"], nlu=nlu, tool_calls=calls, needs_user_input=True)
        try:
            order = self.tools.get_order(order_id, self.user_id)
            calls.append({"tool": "get_order", "status": "success", "order_id": order_id})
            if str(order.get("status", "success")).casefold() in {"error", "failed", "not_found"}:
                return self._result(
                    "I couldn't verify that order. Please check the order ID or contact ShopSphere support.",
                    nlu=nlu, tool_calls=calls, needs_user_input=True,
                )
            state = self.tracker.apply_order_result(order, item_hint=text)
        except Exception as exc:
            calls.append({"tool": "get_order", "status": "failed", "error_type": type(exc).__name__})
            return self._result(
                "I couldn't retrieve the order details, so I haven't started the pickup. Please try again.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        return self._advance_return_workflow(nlu, calls)

    def _order_status(self, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        order_ids = (nlu.get("entities") or {}).get("order_ids") or []
        if not order_ids:
            return self._result("Please share the order ID so I can look it up.", nlu=nlu, tool_calls=calls, needs_user_input=True)
        try:
            order = self.tools.get_order(order_ids[0], self.user_id)
            calls.append({"tool": "get_order", "status": "success", "order_id": order_ids[0]})
            return self._result(
                f"Order {order.get('order_id', order_ids[0])} status: "
                f"{order.get('order_status', 'Status not supplied by the order service')}.",
                nlu=nlu, tool_calls=calls,
            )
        except Exception as exc:
            calls.append({"tool": "get_order", "status": "failed", "error_type": type(exc).__name__})
            return self._result("I couldn't retrieve that order's status. Please try again.", nlu=nlu, tool_calls=calls, needs_user_input=True)

    @staticmethod
    def _is_historical_policy_query(text: str) -> bool:
        """Return True when the customer explicitly asks about an older policy."""
        return bool(re.search(
            r"\b(?:202[0-5]|old|older|previous|prior|past|historical|archived|superseded)\b",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _is_policy_query(text: str) -> bool:
        """Identify domain-specific policy questions that default to the active version."""
        return bool(re.search(
            r"\b(?:return|refund|shipping|delivery|exchange|cancellation|warranty)\s+"
            r"(?:policy|policies|rules|window|period)\b"
            r"|\b(?:policy|policies)\s+(?:for|on)\s+"
            r"(?:returns?|refunds?|shipping|delivery|exchanges?|cancellations?|warranty)\b",
            text,
            re.IGNORECASE,
        ))

    @staticmethod
    def _resolve_policy_status(hit: Mapping[str, Any]) -> str:
        """Resolve policy status from explicit metadata, heading text, then this KB's manifest IDs.

        The real retrieval gateway should return status metadata. The heading
        and known-document fallbacks keep lightweight test adapters compatible
        while remaining strict for unknown document IDs.
        """
        # 1. Prefer explicit top-level or nested metadata.
        metadata = hit.get("metadata") or hit.get("meta") or {}
        for source in (hit, metadata):
            if isinstance(source, Mapping):
                for key in ("status", "document_status", "policy_status"):
                    raw_status = str(source.get(key, "")).strip().casefold()
                    if raw_status:
                        if raw_status in {"active", "current", "published", "in_force"}:
                            return "active"
                        if raw_status in {"superseded", "archived", "obsolete", "retired", "expired"}:
                            return "superseded"
                        # An explicit but unfamiliar status must not be
                        # overridden by a document-ID guess.
                        return raw_status

        # 2. Read the source heading/content when a status field is absent.
        descriptive_text = " ".join(
            str(hit.get(key, "") or "")
            for key in ("section", "title", "content")
        )
        import re
        status_match = re.search(
            r"\bstatus\s*:\s*(active|current|superseded|archived)\b",
            descriptive_text,
            re.IGNORECASE,
        )
        if status_match:
            value = status_match.group(1).casefold()
            return "active" if value in {"active", "current"} else "superseded"

        if re.search(r"\b(?:current|active)\b", str(hit.get("section", "") or ""), re.IGNORECASE):
            return "active"
        if re.search(r"\b(?:superseded|archived)\b", str(hit.get("section", "") or ""), re.IGNORECASE):
            return "superseded"

        # 3. Last-resort fallback from the project's authoritative version-pair
        # manifest. KB-001/KB-002 are the returns pair; KB-016/KB-005 are shipping.
        doc_id = str(hit.get("doc_id") or hit.get("document_id") or "").strip().upper()
        manifest_status = {
            "KB-001": "superseded",
            "KB-002": "active",
            "KB-005": "active",
            "KB-016": "superseded",
        }
        return manifest_status.get(doc_id, "unknown")

    def _knowledge_base_answer(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        # Current-policy questions need a wider candidate pool because archived
        # versions can otherwise occupy most of a small top-k result list.
        current_policy_query = (
            self._is_policy_query(text)
            and not self._is_historical_policy_query(text)
        )
        search_top_k = max(self.top_k, 8) if current_policy_query else self.top_k

        retrieval_started = timer()
        print("[RecallBot] Retrieval started")
        try:
            candidate_hits = list(self.tools.search_kb(text, top_k=search_top_k))
            print(f"[RecallBot] Retrieval completed: {(timer()-retrieval_started)*1000:.0f} ms")
        except Exception as exc:
            calls.append({"tool": "search_kb", "status": "failed", "error_type": type(exc).__name__})
            return self._result(
                "I couldn't access the policy knowledge base just now. Please try again.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )

        hits = candidate_hits
        version_filter = "none"

        policy_started = timer()
        if current_policy_query:
            # Only metadata-marked active/current documents can support a
            # present-day policy answer. Never silently promote archived text.
            active_hits = []
            for hit in candidate_hits:
                resolved_status = self._resolve_policy_status(hit)
                if resolved_status in {"active", "current"}:
                    # Enrich metadata-incomplete adapters/test doubles with the
                    # status resolved from the authoritative project KB manifest.
                    enriched_hit = dict(hit)
                    enriched_hit.setdefault("status", resolved_status)
                    active_hits.append(enriched_hit)
            if not active_hits:
                calls.append({
                    "tool": "search_kb",
                    "status": "success",
                    "candidate_count": len(candidate_hits),
                    "result_count": 0,
                    "version_filter": "active_only",
                })
                return self._result(
                    "I found policy-related material, but none of the retrieved passages is marked as the active policy. "
                    "I don't want to apply an archived rule. Please provide the order delivery date or contact ShopSphere support.",
                    nlu=nlu, tool_calls=calls, needs_user_input=True,
                )
            hits = active_hits[:self.top_k]
            version_filter = "active_only"
        print(f"[RecallBot] Policy-version resolution completed: {(timer()-policy_started)*1000:.0f} ms")

        calls.append({
            "tool": "search_kb",
            "status": "success",
            "candidate_count": len(candidate_hits),
            "result_count": len(hits),
            "version_filter": version_filter,
        })

        if not hits:
            return self._result(
                "I couldn't find a supporting policy passage, so I don't want to guess. Please contact ShopSphere support.",
                nlu=nlu, tool_calls=calls,
            )

        if self.response_generator is not None:
            reply = self.response_generator(text, hits)
        else:
            # Keep a useful deterministic fallback if Ollama is unavailable.
            # When Qwen is enabled, _result() replaces this draft with concise
            # instructions and supplies the selected passages separately.
            first = hits[0]
            content = str(first.get("content", "")).strip()
            source_labels = [self._source_label(hit) for hit in hits]
            source_text = "; ".join(source_labels)
            reply = f"The retrieved policy information says: {content}"
            if source_text:
                reply += f"\\n\\nSources: {source_text}"

        return self._result(
            reply,
            nlu=nlu,
            tool_calls=calls,
            sources=[self._source_label(hit) for hit in hits],
            nlg_evidence=hits,
        )

    def _advance_return_workflow(self, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.tracker.state
        if state.get("clarification_required"):
            return self._result(
                state.get("clarification_prompt") or "Which item would you like to return?",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )

        # Never prepare a second booking action while one is already awaiting
        # the customer's response. This is a defensive guard even when NLU
        # cannot classify an unusual yes/no utterance.
        if state.get("pending_confirmation"):
            return self._result(
                "A pickup booking is already awaiting your confirmation. "
                "Please reply yes to confirm it or no to cancel it.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )

        missing = self.tracker.missing_slots(include_confirmation=False)
        if missing:
            slot = missing[0]
            prompt = _SLOT_PROMPTS.get(slot, f"Please provide {slot.replace('_', ' ')}.")
            return self._result(prompt, nlu=nlu, tool_calls=calls, needs_user_input=True)

        slots = state["slots"]
        try:
            requested_window = slots["pickup_time_window"]
            available = list(self.tools.check_pickup_slots(
                pickup_date=slots["pickup_date"],
                requested_window=requested_window,
                address_summary=slots["pickup_address_summary"],
            ))

            # Some slot gateways may return all windows after an "after 6 pm"
            # request. If the user subsequently selects an exact range such as
            # 18:00–20:00, enforce that selection here as well.
            requested_start = requested_window.get("start")
            requested_end = requested_window.get("end")
            if requested_start and requested_end:
                exact_matches = [
                    slot for slot in available
                    if str(slot.get("window_start", "")).strip() == str(requested_start).strip()
                    and str(slot.get("window_end", "")).strip() == str(requested_end).strip()
                ]
                available = exact_matches

            calls.append({"tool": "check_pickup_slots", "status": "success", "result_count": len(available)})
        except Exception as exc:
            calls.append({"tool": "check_pickup_slots", "status": "failed", "error_type": type(exc).__name__})
            return self._result("I couldn't verify pickup availability. No booking has been made. Please try again.", nlu=nlu, tool_calls=calls, needs_user_input=True)

        if not available:
            return self._result(
                "That pickup window is not currently available. Please choose another date or time window.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        if len(available) > 1:
            options = [self._format_slot(slot) for slot in available]
            reply = "More than one pickup slot is available. Please choose one:\n" + "\n".join(f"- {option}" for option in options)
            return self._result(reply, nlu=nlu, tool_calls=calls, needs_user_input=True)

        selected = available[0]
        selected_window = {
            "start": str(selected.get("window_start")),
            "end": str(selected.get("window_end")),
        }
        self.tracker.set_slot("pickup_time_window", selected_window)
        action_arguments = {
            "customer_id": slots["customer_id"],
            "order_id": slots["order_id"],
            "sku": slots["sku"],
            "reason": slots["reason"],
            "pickup_date": slots["pickup_date"],
            "pickup_time_window": selected_window,
            "pickup_address_summary": slots["pickup_address_summary"],
            "slot_id": selected.get("slot_id"),
        }
        readiness = self.tracker.request_confirmation({"tool": "book_slot", "arguments": action_arguments})
        if not readiness.get("ready"):
            return self._result(
                "I still need some information before I can prepare the booking: "
                + ", ".join(readiness.get("missing_slots", [])),
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        return self._result(
            self._render_booking_summary(action_arguments)
            + "\n\nWould you like me to confirm this pickup booking? Please reply yes or no.",
            nlu=nlu, tool_calls=calls, needs_user_input=True,
        )

    def _handle_confirmation(self, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if self.tracker.state.get("confirmation_mismatch"):
            return self._result(
                "The order or item in that confirmation does not match the pending pickup. "
                "Please confirm the displayed order and item, or cancel the pending booking.",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        action = self.tracker.consume_confirmed_action()
        if action is None:
            return self._result(
                "There is no pending booking awaiting confirmation, so I haven't made any database change.",
                nlu=nlu, tool_calls=calls, needs_user_input=False,
            )
        if action.get("tool") != "book_slot":
            self.tracker.record_action_failure("unsupported_tool")
            return self._result("The confirmed action is not supported by the current tool gateway.", nlu=nlu, tool_calls=calls)
        try:
            # The tracker only releases this action after an explicit user
            # confirmation turn. Pass that verified fact to the tool adapter,
            # which independently rejects writes without this flag.
            confirmed_arguments = dict(action.get("arguments", {}))
            confirmed_arguments["explicit_confirmation"] = True
            result = self.tools.book_slot(confirmed_arguments)
            calls.append({"tool": "book_slot", "status": str(result.get("status", "unknown"))})
            if str(result.get("status", "")).casefold() not in {"success", "booked", "confirmed"}:
                self.tracker.record_action_failure(str(result.get("error_type", "booking_not_confirmed")))
                return self._result(
                    "The booking service did not confirm the pickup. No successful booking is being reported.",
                    nlu=nlu, tool_calls=calls,
                )
            self.tracker.record_action_result(result)
            reference = result.get("rma_id") or result.get("booking_id") or result.get("pickup_id")
            status = result.get("booking_status") or result.get("status")
            if not reference:
                return self._result(
                    "The booking service returned a success status but no booking reference. "
                    "Please check the booking record before relying on a reference.",
                    nlu=nlu, tool_calls=calls,
                )
            return self._result(
                f"Your pickup has been booked. Reference: {reference}. Status: {status}.",
                nlu=nlu, tool_calls=calls, completed=True,
            )
        except Exception as exc:
            self.tracker.record_action_failure(type(exc).__name__)
            calls.append({"tool": "book_slot", "status": "failed", "error_type": type(exc).__name__})
            return self._result("The booking attempt failed. I haven't received a confirmation from the booking service.", nlu=nlu, tool_calls=calls)

    @staticmethod
    def _format_slot(slot: Mapping[str, Any]) -> str:
        return f"{slot.get('window_start')}–{slot.get('window_end')} (slot {slot.get('slot_id', 'unavailable')})"

    @staticmethod
    def _render_pickup_summary(state: Mapping[str, Any]) -> str:
        slots = state.get("slots", {})
        return (
            "Please review the return pickup details:\n"
            f"- Order: {slots.get('order_id')}\n"
            f"- Item SKU: {slots.get('sku')}\n"
            f"- Reason: {slots.get('reason')}\n"
            f"- Pickup date: {slots.get('pickup_date')}\n"
            f"- Pickup window: {slots.get('pickup_time_window')}\n"
            f"- Address: {slots.get('pickup_address_summary')}"
        )

    @staticmethod
    def _render_booking_summary(arguments: Mapping[str, Any]) -> str:
        """Render the exact payload held by the pending booking action."""
        return (
            "Please review the return pickup details:\n"
            f"- Order: {arguments.get('order_id')}\n"
            f"- Item SKU: {arguments.get('sku')}\n"
            f"- Reason: {arguments.get('reason')}\n"
            f"- Pickup date: {arguments.get('pickup_date')}\n"
            f"- Pickup window: {arguments.get('pickup_time_window')}\n"
            f"- Slot: {arguments.get('slot_id')}\n"
            f"- Address: {arguments.get('pickup_address_summary')}"
        )

    @staticmethod
    def _source_label(hit: Mapping[str, Any]) -> str:
        doc_id = hit.get("doc_id") or hit.get("document_id") or "unknown-document"
        chunk_id = hit.get("chunk_id") or "unknown-chunk"
        version = hit.get("version")
        label = f"{doc_id} / {chunk_id}"
        return f"{label} (version {version})" if version else label

    def _result(
        self,
        reply: str,
        *,
        nlu: Mapping[str, Any],
        tool_calls: list[dict[str, Any]],
        needs_user_input: bool = False,
        completed: bool = False,
        sources: list[str] | None = None,
        nlg_evidence: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a turn result and, when configured, render it through local Qwen.

        The Dialogue Manager remains authoritative for actions and state. Qwen
        only rewrites the deterministic response draft using the current turn,
        state and supplied retrieval evidence.
        """
        state_snapshot = self.tracker.snapshot()
        original_reply = reply
        nlg_status = "disabled"
        nlg_error = None
        if self.nlg is not None:
            intent = str(nlu.get("intent", "unknown"))
            turn_type = "open" if intent in {"greeting", "small_talk", "closing"} else "task"
            try:
                # Do not duplicate a full retrieved passage in both the draft
                # and the evidence payload. Keep the deterministic extractive
                # reply as the offline fallback, but give Qwen a concise task
                # instruction and the actual retrieved passages separately.
                nlg_draft = original_reply
                if nlg_evidence:
                    nlg_draft = (
                        "Answer the customer's policy question using only the supplied retrieved passages. "
                        "For a current-policy question, summarise only the active policy. "
                        "Give a complete answer in no more than about 90 words; use a few short bullets if helpful. "
                        "Include the effective date when relevant. If a requested detail is absent, say it is not specified. "
                        "Do not write a Sources or Citations footer; the application adds source metadata."
                    )
                reply = self.nlg.generate_reply(
                    nlg_draft,
                    user_message=(
                        self.rolling_summary_memory.redact_text(
                            str(state_snapshot.get("last_user_text") or "")
                        )
                        if self.rolling_summary_memory is not None
                        else str(state_snapshot.get("last_user_text") or "")
                    ),
                    intent=intent,
                    state=state_snapshot,
                    evidence=nlg_evidence,
                    sources=sources or [],
                    turn_type=turn_type,
                    conversation_history=getattr(self, "_active_conversation_history", []),
                    conversation_summary=getattr(self, "_active_conversation_summary", ""),
                )
                nlg_status = "qwen"
            except Exception as exc:
                # Preserve the deterministic response if Qwen NLG fails.
                reply = original_reply

                # Keep the status stable for existing callers and tests.
                nlg_status = f"fallback:{type(exc).__name__}"

                # Store diagnostic details separately from the stable status.
                nlg_error = " ".join(str(exc).split())[:180] or None

        return {
            "reply": reply,
            "intent": nlu.get("intent", "unknown"),
            "nlu": dict(nlu),
            "state": state_snapshot,
            "tool_calls": tool_calls,
            "sources": sources or [],
            "needs_user_input": needs_user_input,
            "completed": completed,
            "nlg_status": nlg_status,
            "nlg_error": nlg_error,
        }
