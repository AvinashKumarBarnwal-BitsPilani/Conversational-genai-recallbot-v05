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
from ..config import VALID_CONFIGS
from ..logger import log_call, timer
from ..context.context_assembler import FALLBACK_SENTENCE, ContextAssembler, assembly_record, build_turn_items
from ..memory.fact_extractor import is_fact_like
from ..memory.forgetting import resolve_fact_keys
from ..memory.memory_manager import (
    EPISODE_MIN_RELEVANCE_CONTEXT, EPISODE_MIN_RELEVANCE_QUERY, FACT_MIN_SCORE_CONTEXT, FACT_MIN_SCORE_QUERY,
)
from ..memory.records import key_label
from .qwen_nlg import DECODING_PRESETS, _compact_state

TASK4_CONFIGS = frozenset({"no_rag", "hybrid", "hybrid_sem", "hybrid_mem"})
# Turns that never need long-term memory retrieval (saves embedding calls).
NO_RETRIEVAL_INTENTS = frozenset({
    "greeting", "small_talk", "closing", "confirm_action", "cancel_action", "forget_request",
    "forgotten_memory_query", "provide_user_fact", "update_user_fact",
})
MEMORY_QUERY_INTENTS = frozenset({"semantic_memory_query", "episodic_memory_query"})


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
        memory: Any | None = None,
    ) -> None:
        self.tracker = tracker
        self.tools = tools
        self.user_id = user_id or tracker.state.get("user_id")
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {sorted(VALID_CONFIGS)}")
        # Task 2 configs name the history strategy itself. Task 4 configs use
        # the V05 trim window (k=3) for short-term history.
        selected_strategy = memory_strategy or (config if config in {"full", "trim", "summary"} else "trim")
        if selected_strategy not in {"full", "trim", "summary"}:
            raise ValueError("memory_strategy must be 'full', 'trim' or 'summary'")
        self.memory_strategy = selected_strategy
        self.config = config if config not in {"full", "trim", "summary"} else selected_strategy
        # Long-term memory (Task 3): a MemoryManager, or None for no persistence.
        self.memory = memory
        # Task 4 configs assemble every prompt under the V05 budget with the
        # compact versioned prompt; Task 2 configs keep their original builder.
        self.use_context_assembler = self.config in TASK4_CONFIGS
        self.context_assembler = ContextAssembler() if self.use_context_assembler else None
        self.system_prompt_v2 = (
            (Path(__file__).resolve().parents[2] / "skills" / "system_prompt_v2.txt").read_text(encoding="utf-8").strip()
            if self.use_context_assembler else ""
        )
        self._turn_memory: dict[str, list[Any]] = {"facts": [], "episodes": []}
        self.last_context_assembly: dict[str, Any] | None = None
        self._previous_context_assembly: dict[str, Any] | None = None
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
        if self.memory is not None:
            self.memory.set_call_context(**call_context)

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
        self._previous_context_assembly = self.last_context_assembly
        self.last_context_assembly = None
        self._turn_memory = {"facts": [], "episodes": []}
        if self.memory is not None and intent not in NO_RETRIEVAL_INTENTS:
            memory_query = intent in MEMORY_QUERY_INTENTS
            try:
                self._turn_memory = self.memory.retrieve_for_turn(
                    text, fact_top_k=3 if memory_query else 2, episode_top_k=3 if memory_query else 2,
                    fact_min_score=FACT_MIN_SCORE_QUERY if memory_query else FACT_MIN_SCORE_CONTEXT,
                    episode_min_relevance=(EPISODE_MIN_RELEVANCE_QUERY if memory_query
                                           else EPISODE_MIN_RELEVANCE_CONTEXT),
                )
                calls.append({"tool": "memory_search", "status": "success",
                              "facts": [f.memory_id for f, _ in self._turn_memory["facts"]],
                              "episodes": [s.episode.episode_id for s in self._turn_memory["episodes"]]})
            except Exception as exc:
                calls.append({"tool": "memory_search", "status": "failed", "error_type": type(exc).__name__})

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
            if self.use_context_assembler:
                reserve = int(DECODING_PRESETS["task"]["num_predict"])
                budget = self.context_assembler.context_window - reserve - self.context_assembler.safety_margin
                previous = self._previous_context_assembly
                last_turn = ""
                if previous:
                    cut = len(previous.get("dropped", [])) + len(previous.get("dropped_by_trim", []))
                    last_turn = (
                        f" On the previous turn the prompt used {previous['total_input_tokens']:,} of "
                        f"{previous['input_budget']:,} input tokens and {cut} item(s) were cut."
                    )
                # Verbatim (pilot/official P15): the model distorted a correct draft
                # about the system's own configuration, which it cannot improve on.
                return self._result(
                    f"My context window is {self.context_assembler.context_window:,} tokens. I reserve {reserve} "
                    f"for my reply and {self.context_assembler.safety_margin} as a safety margin, so each prompt "
                    f"gets at most {budget:,} input tokens, counted with the Qwen tokenizer. The instructions, your "
                    "current request and the task state always go in; saved facts, past events and knowledge-base "
                    "chunks are added by relevance; conversation turns older than the last 3 and any "
                    "lower-priority item that does not fit are cut, and everything cut is recorded." + last_turn,
                    nlu=nlu, tool_calls=calls, verbatim=True,
                )
            return self._result(
                "This RecallBot variant uses a configured 2,048-token input context budget. "
                "The context manager will select the information included in each prompt.",
                nlu=nlu, tool_calls=calls,
            )
        if intent == "forgotten_memory_query":
            return self._answer_forgotten(text, nlu, calls)
        if intent == "forget_request":
            return self._handle_forget(text, nlu, calls)
        if intent == "report_complaint":
            return self._handle_complaint(text, nlu, calls)
        if intent == "report_decision":
            return self._handle_decision(text, nlu, calls)
        if intent == "semantic_memory_query":
            return self._answer_semantic(text, nlu, calls)
        if intent == "episodic_memory_query":
            return self._answer_episodic(nlu, calls)

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
                # slots and availability. For an open-ended window ("after 6 pm")
                # propose the earliest satisfying slot for explicit confirmation;
                # bounded windows with several matches still require a choice.
                return self._advance_return_workflow(
                    nlu, calls, propose_earliest_for_open_window=True
                )
            return self._advance_return_workflow(nlu, calls)

        # A state-follow-up can also ask about the general process after a
        # referenced workflow stage (for example, what happens after pickup).
        # If this session has no active return workflow, answer from retrieved
        # KB evidence; do not manufacture Session A state in a new session.
        if intent == "state_followup":
            return self._knowledge_base_answer(text, nlu, calls)

        if intent in {"provide_user_fact", "update_user_fact"} or is_fact_like(text, nlu):
            return self._handle_user_fact(text, nlu, calls)

        return self._result(
            "I can help with an order, return, delivery or policy question. "
            "Could you tell me a little more about what you need?",
            nlu=nlu, tool_calls=calls, needs_user_input=True,
        )

    # ------------------------------------------------------------------
    # Long-term memory writes (Task 3). The MemoryManager decides, per
    # config, whether semantic/episodic memory is enabled at all.
    # ------------------------------------------------------------------
    def _semantic_on(self) -> bool:
        return self.memory is not None and bool(getattr(self.memory, "semantic_enabled", False))

    def _episodic_on(self) -> bool:
        return self.memory is not None and bool(getattr(self.memory, "episodic_enabled", False))

    def _remember(
        self, calls: list[dict[str, Any]], episode_type: str, description: str, **kwargs: Any
    ) -> Any | None:
        """Write one episode; a memory failure is reported, never hidden or fatal."""
        if not self._episodic_on():
            return None
        try:
            episode = self.memory.record_episode(episode_type, description, **kwargs)
        except Exception as exc:
            calls.append({"tool": "episodic_memory_write", "status": "failed", "error_type": type(exc).__name__})
            return None
        if episode is not None:
            calls.append({"tool": "episodic_memory_write", "status": "success",
                          "episode_id": episode.episode_id, "type": episode_type})
        return episode

    # ------------------------------------------------------------------
    # Long-term memory reads. Decisions are deterministic; when nothing
    # relevant clears its threshold the exact fallback sentence is returned
    # without the model, so it cannot be paraphrased into a guess.
    # ------------------------------------------------------------------
    def _no_memory_answer(self, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        """Configs without that memory type may still answer from short-term history."""
        return self._result(
            "Answer only if the conversation history above contains the answer; otherwise reply exactly: "
            + FALLBACK_SENTENCE,
            nlu=nlu, tool_calls=calls, fallback_reply=FALLBACK_SENTENCE,
        )

    def _answer_semantic(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._semantic_on():
            return self._no_memory_answer(nlu, calls)
        hits = list(self._turn_memory["facts"])
        named_keys = resolve_fact_keys(text, nlu)
        if named_keys:
            # The customer named the detail: answer from that key, never a neighbour.
            named = [self.memory.semantic.get_active(self.user_id, key) for key in named_keys]
            hits = [(fact, 1.0) for fact in named if fact is not None]
            if not hits and any(self.memory.tombstone_for(key) for key in named_keys):
                return self._answer_forgotten(text, nlu, calls)
            self._turn_memory["facts"] = hits + [h for h in self._turn_memory["facts"]
                                                 if h[0].memory_id not in {f.memory_id for f, _ in hits}]
        if not hits:
            return self._result(FALLBACK_SENTENCE, nlu=nlu, tool_calls=calls, verbatim=True)
        fact = hits[0][0]
        return self._result(
            f"Your saved {key_label(fact.key)} is {fact.value} [{fact.memory_id}], saved on {fact.date[:10]}.",
            nlu=nlu, tool_calls=calls,
        )

    def _answer_episodic(self, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._episodic_on():
            return self._no_memory_answer(nlu, calls)
        scored = list(self._turn_memory["episodes"])
        if not scored:
            return self._result(FALLBACK_SENTENCE, nlu=nlu, tool_calls=calls, verbatim=True)
        top = scored[0].episode
        return self._result(
            f"From your history on {top.timestamp[:10]} [{top.episode_id}]: {top.description}",
            nlu=nlu, tool_calls=calls,
        )

    def _answer_forgotten(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        keys = resolve_fact_keys(text, nlu)
        label = " and ".join(key_label(key) for key in keys) or "detail"
        forgotten = self.memory is not None and any(self.memory.tombstone_for(key) for key in keys)
        reply = (
            f"I no longer keep your {label}. It was deleted at your request, so I can't tell you what it was."
            if forgotten else f"I don't have a saved {label}, so there is nothing I can tell you about it."
        )
        # Verbatim: a deleted value must never be regenerated by the model.
        return self._result(reply, nlu=nlu, tool_calls=calls, verbatim=True)

    def _handle_user_fact(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._semantic_on():
            return self._result(
                "Thanks, I've noted that for this conversation. Long-term memory is not enabled in this "
                "configuration, so it won't be saved for future sessions.",
                nlu=nlu, tool_calls=calls,
            )
        try:
            outcome = self.memory.observe_user_turn(text, nlu)
        except Exception as exc:
            calls.append({"tool": "semantic_memory_write", "status": "failed", "error_type": type(exc).__name__})
            return self._result(
                "I couldn't save that detail just now, so I won't claim it has been stored.",
                nlu=nlu, tool_calls=calls,
            )
        sentences: list[str] = []
        for item in outcome["actions"]:
            record = item.get("record") or {}
            label, value = key_label(str(record.get("key", ""))), record.get("value")
            calls.append({"tool": "semantic_memory_write", "status": "success",
                          "action": item["action"], "memory_id": record.get("memory_id")})
            if item["action"] == "created":
                sentences.append(f"I've saved your {label}: {value}.")
            elif item["action"] == "updated":
                sentences.append(f"I've updated your {label} from {item['superseded']['value']} to {value}.")
            else:
                sentences.append(f"I already have your {label} saved as {value}.")
        if not sentences:
            return self._result(
                "I couldn't identify a specific preference to save from that. Could you rephrase it?",
                nlu=nlu, tool_calls=calls, needs_user_input=True,
            )
        return self._result("Thanks. " + " ".join(sentences), nlu=nlu, tool_calls=calls)

    def _handle_forget(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._semantic_on():
            return self._result(
                "Long-term memory is not enabled in this configuration, so no saved details are kept "
                "beyond this conversation.",
                nlu=nlu, tool_calls=calls,
            )
        try:
            report = self.memory.forget(text, nlu)
        except Exception as exc:
            calls.append({"tool": "memory_forget", "status": "failed", "error_type": type(exc).__name__})
            return self._result(
                "I couldn't complete the deletion just now, so I won't claim it has been forgotten.",
                nlu=nlu, tool_calls=calls,
            )
        calls.append({"tool": "memory_forget", "status": "success", **report})
        if not report["keys"]:
            return self._result("Which saved detail would you like me to forget?", nlu=nlu,
                                tool_calls=calls, needs_user_input=True)
        labels = " and ".join(key_label(key) for key in report["keys"])
        if not report["facts_removed"]:
            return self._result(f"I don't have a saved {labels} to forget.", nlu=nlu, tool_calls=calls)
        return self._result(
            f"Done. I've deleted your saved {labels} from my records, including any history that mentioned it.",
            nlu=nlu, tool_calls=calls,
        )

    def _handle_complaint(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        order_ids = (nlu.get("entities") or {}).get("order_ids") or []
        issue = "late delivery" if re.search(r"\b(late|delay(?:ed)?|missed)\b", text, re.I) else "delivery problem"
        order_note = ""
        if order_ids:
            try:
                order = self.tools.get_order(order_ids[0], self.user_id)
                calls.append({"tool": "get_order", "status": "success", "order_id": order_ids[0]})
                order_note = f" (order status {order.get('order_status', 'unknown')})"
            except Exception as exc:
                calls.append({"tool": "get_order", "status": "failed", "error_type": type(exc).__name__})
                return self._result(
                    "I couldn't verify that order, so I haven't recorded the issue against it. "
                    "Please check the order ID.",
                    nlu=nlu, tool_calls=calls, needs_user_input=True,
                )
        subject = f"order {order_ids[0]}" if order_ids else "an order"
        episode = self._remember(
            calls, "complaint",
            f"Customer reported a {issue} for {subject}{order_note}. Recorded as a customer-reported issue; "
            "no formal complaint case was created.",
            entities={"order_id": order_ids[0] if order_ids else None, "issue": issue},
        )
        noted = " I've noted this issue in your history." if episode is not None else ""
        return self._result(
            f"I'm sorry about the {issue} on {subject}.{noted} I can't open a formal complaint case from here, "
            "so no case reference has been created.",
            nlu=nlu, tool_calls=calls,
        )

    def _handle_decision(self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        order_ids = (nlu.get("entities") or {}).get("order_ids") or []
        subject = f"order {order_ids[0]}" if order_ids else "an earlier order"
        match = re.search(r"\b(exchange|refund)\s+(?:rather than|instead of)\s+(?:an?\s+)?(exchange|refund)\b", text, re.I)
        if match:
            decision = f"chose {match.group(1).lower()} rather than {match.group(2).lower()}"
        else:
            decision = "reported a decision"
        episode = self._remember(
            calls, "decision", f"Customer {decision} for {subject}.",
            entities={"order_id": order_ids[0] if order_ids else None, "decision": decision},
        )
        noted = " I've noted that in your history." if episode is not None else ""
        return self._result(
            f"Thanks for letting me know that you {decision} for {subject}.{noted} "
            "I won't apply that choice to your current return unless you ask.",
            nlu=nlu, tool_calls=calls,
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
        sku = state["slots"].get("sku")
        item_name = next((item.get("name") for item in state.get("available_order_items", [])
                          if item.get("sku") == sku), None)
        self._remember(
            calls, "task_event",
            f"Customer started a return request for {item_name or 'an item'}"
            f"{f' ({sku})' if sku else ''} in order {order_id}.",
            entities={"order_id": order_id, "sku": sku},
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
        if self.config == "no_rag":
            # Ablation: short-term memory only, no knowledge-base retrieval.
            return self._result(
                "No knowledge-base passages are available in this configuration. Answer only if the "
                "conversation history above contains the answer; otherwise reply exactly: " + FALLBACK_SENTENCE,
                nlu=nlu, tool_calls=calls, fallback_reply=FALLBACK_SENTENCE,
            )
        return self._knowledge_base_answer_with_retrieval(text, nlu, calls)

    def _knowledge_base_answer_with_retrieval(
        self, text: str, nlu: Mapping[str, Any], calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Current-policy questions need a wider candidate pool because archived
        # versions can otherwise occupy most of a small top-k result list.
        current_policy_query = (
            self._is_policy_query(text)
            and not self._is_historical_policy_query(text)
        )
        # The wider pool also lets the superseded-chunk filter keep top_k results.
        search_top_k = (max(self.top_k, 8) if current_policy_query or self.use_context_assembler
                        else self.top_k)

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

        hits = candidate_hits[:self.top_k]
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
        elif self.use_context_assembler and not self._is_historical_policy_query(text):
            # Task 4 configs: a superseded chunk must not enter a 2,048-token
            # prompt unless the customer asks about an older period (the CS4
            # stale-chunk failure seen in the pilot at turn 31).
            current = [hit for hit in candidate_hits if self._resolve_policy_status(hit) != "superseded"]
            if current and len(current) < len(candidate_hits):
                hits = current[:self.top_k]
                version_filter = "exclude_superseded"
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

    def _advance_return_workflow(
        self,
        nlu: Mapping[str, Any],
        calls: list[dict[str, Any]],
        *,
        propose_earliest_for_open_window: bool = False,
    ) -> dict[str, Any]:
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
        alternatives: list[Mapping[str, Any]] = []
        open_ended_window = bool(requested_start) and not requested_end
        if len(available) > 1 and propose_earliest_for_open_window and open_ended_window:
            # "After 6 pm" is satisfied by every returned slot; the earliest one
            # is the closest match. It is only proposed: the write still needs
            # an explicit confirmation turn and live revalidation.
            ordered = sorted(available, key=lambda slot: str(slot.get("window_start", "")))
            available, alternatives = ordered[:1], ordered[1:]
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
        alternative_text = ""
        if alternatives:
            alternative_text = (
                "\n\nThis is the earliest slot matching your requested time. Other available option(s): "
                + "; ".join(self._format_slot(slot) for slot in alternatives)
                + ". Reply no if you prefer one of these instead."
            )
        return self._result(
            self._render_booking_summary(action_arguments)
            + alternative_text
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
            args = confirmed_arguments
            window = args.get("pickup_time_window") or {}
            # Only values returned by the real booking tool are recorded.
            self._remember(
                calls, "booking",
                f"Return pickup booked for {args.get('sku')} in order {args.get('order_id')} on "
                f"{args.get('pickup_date')}, {window.get('start')}-{window.get('end')} "
                f"(slot {result.get('slot_id') or args.get('slot_id')}); reason {args.get('reason')}; "
                f"RMA {result.get('rma_id')}, booking {result.get('booking_id')}, status {status}.",
                entities={key: result.get(key) or args.get(key) for key in
                          ("order_id", "sku", "slot_id", "rma_id", "booking_id", "pickup_date")},
            )
            return self._result(
                f"Your pickup has been booked. Reference: {reference}. Status: {status}.",
                nlu=nlu, tool_calls=calls, completed=True,
            )
        except Exception as exc:
            self.tracker.record_action_failure(type(exc).__name__)
            calls.append({"tool": "book_slot", "status": "failed", "error_type": type(exc).__name__})
            return self._result("The booking attempt failed. I haven't received a confirmation from the booking service.", nlu=nlu, tool_calls=calls)

    def _history_cut_by_trim(self) -> list[dict[str, Any]]:
        """Completed exchanges the k=3 trim strategy left out of this prompt."""
        if self.memory_strategy != "trim":
            return []
        full = self.conversation_history.get_history("full")
        kept = len(getattr(self, "_active_conversation_history", [])) // 2
        exchanges = [full[i:i + 2] for i in range(0, len(full), 2)]
        cut = exchanges[: max(0, len(exchanges) - kept)]
        return [
            {"id": f"EXCHANGE-{number}", "component": "conversation_history",
             "tokens": self.context_assembler.count("\n".join(m["content"] for m in exchange)),
             "reason": "trim strategy keeps the last 3 exchanges"}
            for number, exchange in enumerate(cut, start=1)
        ]

    @staticmethod
    def _prompt_state(state: Mapping[str, Any]) -> dict[str, Any]:
        """Dialogue state for the prompt: task fields only while a task is active.

        In the pilot, an idle tracker's empty slot list was read out by the
        model as "Missing details: customer_id, order_id ...".
        """
        compact = _compact_state(state)
        if state.get("active_task"):
            return compact
        return {key: compact[key] for key in ("session_id", "turn", "current_intent") if key in compact}

    def _workflow_step(self, state: Mapping[str, Any]) -> str | None:
        """Only the current step of the versioned workflow enters the prompt."""
        if state.get("active_task") != "return_pickup":
            return None
        steps = list(self._workflow.get("steps", []))
        if len(steps) < 11:
            return None
        slots = state.get("slots", {})
        if state.get("task_status") == "completed":
            return steps[10]
        if state.get("pending_confirmation"):
            return steps[8]
        if not slots.get("order_id"):
            return steps[0]
        if not slots.get("sku"):
            return steps[2]
        if not slots.get("reason"):
            return steps[4]
        if not slots.get("pickup_date") or not slots.get("pickup_time_window"):
            return steps[6]
        return steps[7]

    def _generate_with_assembled_context(
        self,
        draft: str,
        deterministic_reply: str,
        *,
        nlu: Mapping[str, Any],
        state: Mapping[str, Any],
        sources: list[str],
        nlg_evidence: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str, str | None]:
        """Assemble the prompt under the 2,048-token budget, then call Qwen."""
        intent = str(nlu.get("intent", "unknown"))
        turn_type = "open" if intent in {"greeting", "small_talk", "closing"} else "task"
        redact = self.memory.redact_text if self.memory is not None else (lambda value: value)
        history = [
            {"role": str(m["role"]), "content": redact(str(m["content"]))}
            for m in getattr(self, "_active_conversation_history", [])
        ]
        nlg_draft = draft
        if nlg_evidence:
            nlg_draft = (
                "Answer the customer's policy question using only the KNOWLEDGE BASE chunks. For a current-policy "
                "question use only active chunks. Cite chunk IDs in brackets. If a detail is absent, say it is not specified."
            )
        try:
            items, profile = build_turn_items(
                intent=intent,
                system_prompt=self.system_prompt_v2,
                system_prompt_version="system_prompt_v2",
                workflow_step=self._workflow_step(state),
                user_message=redact(str(state.get("last_user_text") or "")),
                state=self._prompt_state(state),
                draft=nlg_draft,
                facts=self._turn_memory["facts"],
                episodes=self._turn_memory["episodes"],
                kb_hits=nlg_evidence,
                history=history,
                truncate=self.context_assembler.truncate,
            )
            assembled = self.context_assembler.assemble(
                items, output_reserve=int(DECODING_PRESETS[turn_type]["num_predict"]), profile=profile,
            )
            record = assembly_record(assembled)
            record["prompt_version"] = "system_prompt_v2"
            record["dropped_by_trim"] = self._history_cut_by_trim()
            self.last_context_assembly = record
            reply = self.nlg.generate_reply(
                nlg_draft,
                user_message=redact(str(state.get("last_user_text") or "")),
                intent=intent,
                state=state,
                sources=sources,
                turn_type=turn_type,
                prebuilt_messages=assembled.messages,
                prebuilt_retrieved_tokens=assembled.retrieved_tokens,
                context_assembly={k: v for k, v in record.items() if k != "component_table"},
            )
            return reply, "qwen", None
        except Exception as exc:
            return deterministic_reply, f"fallback:{type(exc).__name__}", " ".join(str(exc).split())[:180] or None

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
        window = arguments.get("pickup_time_window")
        if isinstance(window, Mapping):
            window = f"{window.get('start')}-{window.get('end') or 'open'}"
        arguments = {**arguments, "pickup_time_window": window}
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
        verbatim: bool = False,
        fallback_reply: str | None = None,
    ) -> dict[str, Any]:
        """Build a turn result and, when configured, render it through local Qwen.

        The Dialogue Manager remains authoritative for actions and state. Qwen
        only rewrites the deterministic response draft using the current turn,
        state and supplied retrieval evidence.

        ``verbatim`` replies (the exact fallback sentence, forgotten-fact
        answers) bypass the model. ``fallback_reply`` replaces an instruction
        draft when the model is unavailable.
        """
        state_snapshot = self.tracker.snapshot()
        original_reply = fallback_reply if fallback_reply is not None else reply
        nlg_status = "disabled"
        nlg_error = None
        if verbatim:
            nlg_status = "verbatim"
        elif self.nlg is not None and self.use_context_assembler:
            reply, nlg_status, nlg_error = self._generate_with_assembled_context(
                reply, original_reply, nlu=nlu, state=state_snapshot, sources=sources or [],
                nlg_evidence=nlg_evidence or [],
            )
        elif self.nlg is None:
            reply = original_reply
        if self.nlg is not None and not verbatim and not self.use_context_assembler:
            intent = str(nlu.get("intent", "unknown"))
            turn_type = "open" if intent in {"greeting", "small_talk", "closing"} else "task"
            try:
                # Do not duplicate a full retrieved passage in both the draft
                # and the evidence payload. Keep the deterministic extractive
                # reply as the offline fallback, but give Qwen a concise task
                # instruction and the actual retrieved passages separately.
                nlg_draft = reply
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

        # Last line of defence: no reply may contain a forgotten value.
        if self.memory is not None and self.memory.contains_forgotten(reply):
            reply = self.memory.redact_text(original_reply)
            nlg_status = f"{nlg_status}+forgotten_guard"

        return {
            "context_assembly": self.last_context_assembly,
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
