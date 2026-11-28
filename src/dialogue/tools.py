"""Real JSON-backed tools for RecallBot V05 (Task 2a).

This adapter connects the Dialogue Manager to:
- the existing hybrid retrieval engine (Dense + custom BM25 + RRF);
- the synthetic JSON order database (read-only);
- a persistent JSON pickup-slot database (read and confirmed write).

The seed files are never modified. Runtime booking writes are stored in
``db/runtime_pickup_slots.json`` and protected by a cross-process file lock.
"""
from __future__ import annotations

import json
import os
import tempfile
import hashlib
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import DB_DIR, INDEX_DIR
from ..logger import log_call, timer
from ..retrieval.bm25 import BM25Index
from ..retrieval.dense import DenseIndex
from ..retrieval.rrf import reciprocal_rank_fusion


class ToolError(RuntimeError):
    """Base exception for a failed RecallBot tool operation."""


class OrderNotFound(ToolError):
    pass


class OrderOwnershipError(ToolError):
    pass


class SlotUnavailable(ToolError):
    pass


class ConfirmationRequired(ToolError):
    pass


def _time_to_minutes(value: str | None) -> int | None:
    """Convert a validated HH:MM value into minutes after midnight."""
    if value is None:
        return None
    try:
        hour_text, minute_text = str(value).split(":", maxsplit=1)
        hour, minute = int(hour_text), int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid time value: {value!r}; expected HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: {value!r}; expected HH:MM")
    return hour * 60 + minute


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class RecallBotToolGateway:
    """Concrete tool adapter used by the real Dialogue Manager.

    Parameters are injectable so unit tests can use temporary JSON databases
    and a lightweight fake embedding model without downloading model weights.
    Production defaults use the project's real seed files and retrieval code.
    """

    def __init__(
        self,
        *,
        orders_path: Path | None = None,
        seed_slots_path: Path | None = None,
        runtime_slots_path: Path | None = None,
        index_dir: Path | None = None,
        chunk_size: int = 128,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rrf_k: int = 60,
        candidate_k: int = 20,
        logger: Callable[..., Any] | None = log_call,
        token_counter: Callable[[str], int] | None = None,
        embedder_factory: Callable[[str], Any] | None = None,
        index_loader: Callable[[Path], DenseIndex] | None = None,
    ) -> None:
        self.orders_path = Path(orders_path or (DB_DIR / "seed_orders.json"))
        self.seed_slots_path = Path(seed_slots_path or (DB_DIR / "seed_pickup_slots.json"))
        self.runtime_slots_path = Path(runtime_slots_path or (DB_DIR / "runtime_pickup_slots.json"))
        self.index_dir = Path(index_dir or (INDEX_DIR / f"dense_{chunk_size}"))
        self.chunk_size = int(chunk_size)
        self.embedding_model = embedding_model
        self.rrf_k = int(rrf_k)
        self.candidate_k = int(candidate_k)
        self.logger = logger
        self.token_counter = token_counter
        self.call_history: list[dict[str, Any]] = []
        self.embedder_factory = embedder_factory
        self._custom_index_loader = index_loader is not None
        self.index_loader = index_loader or DenseIndex.load

        self._context: dict[str, Any] = {
            "user": "UNKNOWN", "session": "A", "turn": 0, "config": "full"
        }
        self._context_lock = threading.Lock()
        # Keep lock files in the OS temporary directory rather than the Git
        # repository; the key is stable for the same runtime database path.
        lock_key = hashlib.sha256(str(self.runtime_slots_path.resolve()).encode("utf-8")).hexdigest()[:20]
        self._runtime_lock_path = Path(tempfile.gettempdir()) / f"recallbot-{lock_key}.lock"
        self._dense_index: DenseIndex | None = None
        self._bm25_index: BM25Index | None = None
        self._embedder: Any | None = None
        self._retrieval_lock = threading.Lock()

    # ---------- Per-turn context and mandatory logging ----------

    def set_call_context(
        self, *, user: str, session: str, turn: int, config: str = "full"
    ) -> None:
        """Receive the active dialogue coordinates before tools are invoked."""
        with self._context_lock:
            self._context = {
                "user": str(user or "UNKNOWN"),
                "session": str(session or "A"),
                "turn": int(turn),
                "config": str(config),
            }

    def _count_tokens(self, text: str) -> int:
        if self.token_counter is not None:
            try:
                return max(0, int(self.token_counter(text)))
            except Exception:
                return 0
        try:
            from ..retrieval.tokenizer import count_tokens, load_tokenizer
            # Tool-call accounting runs during a user turn. Never let it
            # trigger a Hugging Face Hub request when the tokenizer is absent.
            tokenizer = load_tokenizer(local_files_only=True)
            return int(count_tokens(text, tokenizer=tokenizer))
        except Exception:
            # Logging should still record the call if the tokenizer is offline.
            return 0

    def _run_logged_tool(
        self,
        *,
        tool_name: str,
        model: str,
        request: Mapping[str, Any],
        operation: Callable[[], Any],
        retrieved_tokens: int = 0,
    ) -> Any:
        """Execute one real tool and log success or failure exactly once."""
        started = timer()
        status = "success"
        error_type: str | None = None
        result: Any = None
        try:
            result = operation()
            return result
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            event: dict[str, Any] = {
                "tool_name": tool_name,
                "status": status,
                "request": deepcopy(dict(request)),
            }
            if status == "success":
                event["result"] = deepcopy(result)
            elif error_type:
                event["error_type"] = error_type
            with self._context_lock:
                self.call_history.append(event)
            if self.logger is not None:
                with self._context_lock:
                    context = dict(self._context)
                extra: dict[str, Any] = {
                    "tool_name": tool_name,
                    "status": status,
                    "request": dict(request),
                }
                if error_type:
                    extra["error_type"] = error_type
                if status == "success":
                    extra["result_summary"] = self._result_summary(result)
                self.logger(
                    config=context["config"],
                    user=context["user"],
                    session=context["session"],
                    turn=context["turn"],
                    call_type="tool",
                    model=model,
                    # The assignment's tool-call convention is zero token
                    # accounting; the logger still records measured latency.
                    input_tokens=0,
                    output_tokens=0,
                    retrieved_tokens=0,
                    start_time=started,
                    extra=extra,
                )

    @staticmethod
    def _result_summary(result: Any) -> dict[str, Any]:
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, dict)):
            return {"result_count": len(result)}
        if isinstance(result, Mapping):
            summary: dict[str, Any] = {}
            for key in ("order_id", "order_status", "slot_id", "booking_id", "rma_id", "status", "booking_status"):
                if key in result:
                    summary[key] = result[key]
            return summary
        return {"result_type": type(result).__name__}

    # ---------- JSON persistence helpers ----------

    @staticmethod
    @contextmanager
    def _exclusive_file_lock(lock_path: Path, timeout_seconds: float = 10.0):
        """Acquire an OS-level lock so separate demo processes cannot double-book."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        started = time.monotonic()
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.seek(0)
                    handle.write(b"0")
                    handle.flush()
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() - started >= timeout_seconds:
                            raise TimeoutError("Timed out waiting for pickup database lock")
                        time.sleep(0.05)
            else:
                import fcntl
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except BlockingIOError:
                        if time.monotonic() - started >= timeout_seconds:
                            raise TimeoutError("Timed out waiting for pickup database lock")
                        time.sleep(0.05)
            yield
        finally:
            try:
                if acquired and os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif acquired:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """Write JSON to a sibling temporary file, then atomically replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise

    def _load_orders(self) -> dict[str, Any]:
        if not self.orders_path.exists():
            raise FileNotFoundError(f"Order database not found: {self.orders_path}")
        payload = json.loads(self.orders_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("orders"), list):
            raise ValueError("Order database must contain an 'orders' array")
        return payload

    def _ensure_runtime_slots(self) -> None:
        if self.runtime_slots_path.exists():
            return
        if not self.seed_slots_path.exists():
            raise FileNotFoundError(f"Pickup-slot seed file not found: {self.seed_slots_path}")
        seed = json.loads(self.seed_slots_path.read_text(encoding="utf-8"))
        if not isinstance(seed.get("slots"), list):
            raise ValueError("Pickup-slot seed database must contain a 'slots' array")
        seed["bookings"] = []
        seed["database_type"] = "runtime_json_pickup_database"
        self._atomic_write_json(self.runtime_slots_path, seed)

    def reset_runtime_database(self) -> None:
        """Reset runtime pickup state from the immutable seed fixture.

        Use only when intentionally starting a fresh demonstration. This does
        not modify the original seed file or the order database.
        """
        if not self.seed_slots_path.exists():
            raise FileNotFoundError(f"Pickup-slot seed file not found: {self.seed_slots_path}")
        seed = json.loads(self.seed_slots_path.read_text(encoding="utf-8"))
        seed["bookings"] = []
        seed["database_type"] = "runtime_json_pickup_database"
        with self._exclusive_file_lock(self._runtime_lock_path):
            self._atomic_write_json(self.runtime_slots_path, seed)

    def _load_runtime_slots_locked(self) -> dict[str, Any]:
        self._ensure_runtime_slots()
        payload = json.loads(self.runtime_slots_path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("slots"), list):
            raise ValueError("Runtime pickup database must contain a 'slots' array")
        payload.setdefault("bookings", [])
        return payload

    # ---------- Tool 1: order database read ----------

    def _find_order(self, order_id: str) -> Mapping[str, Any]:
        payload = self._load_orders()
        order = next(
            (row for row in payload["orders"] if str(row.get("order_id", "")).casefold() == str(order_id).casefold()),
            None,
        )
        if order is None:
            raise OrderNotFound(f"Order {order_id!r} was not found")
        return order

    def get_order(self, order_id: str, customer_id: str | None = None) -> Mapping[str, Any]:
        """Read an order only when the requested customer owns it."""
        request = {"order_id": str(order_id), "customer_id": customer_id}

        def operation() -> Mapping[str, Any]:
            if not order_id:
                raise ValueError("order_id is required")
            if not customer_id:
                raise OrderOwnershipError("A customer identity is required for order lookup")
            order = self._find_order(str(order_id))
            if str(order.get("customer_id", "")).casefold() != str(customer_id).casefold():
                raise OrderOwnershipError("The requested order does not belong to this customer")
            result = deepcopy(dict(order))
            if not result.get("address_summary"):
                # An order without its own address uses the customer's default
                # saved address; it is still shown in the summary for confirmation.
                customer = next((c for c in self._load_orders().get("customers", [])
                                 if str(c.get("customer_id", "")).casefold() == str(customer_id).casefold()), {})
                if customer.get("default_address_summary"):
                    result["address_summary"] = customer["default_address_summary"]
                    result["address_source"] = "customer_default"
            return result

        return self._run_logged_tool(
            tool_name="get_order", model="json-order-database", request=request, operation=operation
        )

    # ---------- Tool 2: hybrid knowledge-base retrieval ----------

    def _load_retrieval_components(self) -> None:
        init_started = timer()
        print("[RecallBot] Retrieval components initialization started")
        with self._retrieval_lock:
            if self._dense_index is None:
                if not self._custom_index_loader and (
                    not (self.index_dir / "chunks.json").exists()
                    or not (self.index_dir / "vectors.npy").exists()
                ):
                    raise FileNotFoundError(
                        f"Hybrid KB index not found at {self.index_dir}. "
                        "Build it first with: python -m src.retrieval.build_index"
                    )
                self._dense_index = self.index_loader(self.index_dir)
                self._bm25_index = BM25Index(self._dense_index.chunks)
            if self._embedder is None:
                if self.embedder_factory is None:
                    from ..retrieval.embeddings import SentenceTransformerEmbedder
                    self._embedder = SentenceTransformerEmbedder(self.embedding_model)
                else:
                    self._embedder = self.embedder_factory(self.embedding_model)
        print(f"[RecallBot] Retrieval components initialization completed: {(timer()-init_started)*1000:.0f} ms")

    def search_kb(self, query: str, top_k: int = 3) -> Sequence[Mapping[str, Any]]:
        """Retrieve and fuse dense and BM25 results using the Day 2 RRF code."""
        request = {"query": str(query), "top_k": int(top_k), "chunk_size": self.chunk_size, "rrf_k": self.rrf_k}
        retrieved_count = 0

        def operation() -> Sequence[Mapping[str, Any]]:
            nonlocal retrieved_count
            if not isinstance(query, str) or not query.strip():
                raise ValueError("A non-empty KB query is required")
            if top_k <= 0:
                return []
            self._load_retrieval_components()
            assert self._dense_index is not None and self._bm25_index is not None and self._embedder is not None
            context = dict(self._context)
            embed_started = timer()
            vectors = self._embedder.encode(
                [query],
                purpose=f"dialogue_search_kb_{self.chunk_size}",
                is_query=True,
                user=context["user"],
                session=context["session"],
                turn=context["turn"],
                config=context["config"],
                show_progress_bar=False,
            )
            print(f"[RecallBot] Query embedding completed: {(timer()-embed_started)*1000:.0f} ms")
            dense_started = timer()
            dense_hits = self._dense_index.search(vectors[0], top_k=self.candidate_k)
            print(f"[RecallBot] Dense retrieval completed: {(timer()-dense_started)*1000:.0f} ms")
            sparse_started = timer()
            bm25_hits = self._bm25_index.search(query, top_k=self.candidate_k)
            dense_ranking = [hit.chunk_id for hit in dense_hits]
            bm25_ranking = [hit.chunk_id for hit in bm25_hits]
            fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=self.rrf_k)
            print(f"[RecallBot] BM25 and RRF completed: {(timer()-sparse_started)*1000:.0f} ms")

            dense_rank_by_id = {hit.chunk_id: hit.rank for hit in dense_hits}
            bm25_rank_by_id = {hit.chunk_id: hit.rank for hit in bm25_hits}
            fused_score_by_id = dict(fused)
            chunk_by_id = {chunk.chunk_id: chunk for chunk in self._dense_index.chunks}
            results: list[dict[str, Any]] = []
            for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1):
                chunk = chunk_by_id[chunk_id]
                results.append({
                    "doc_id": chunk.doc_id,
                    "document_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "version": chunk.version,
                    "effective_from": chunk.effective_from,
                    "effective_to": chunk.effective_to,
                    "status": chunk.status,
                    "source_file": chunk.source_file,
                    "content": chunk.content,
                    "score": float(score),
                    "rank": rank,
                    "method": "hybrid_rrf",
                    "dense_rank": dense_rank_by_id.get(chunk_id),
                    "bm25_rank": bm25_rank_by_id.get(chunk_id),
                    "rrf_k": self.rrf_k,
                    "chunk_size": self.chunk_size,
                })
            retrieved_count = sum(self._count_tokens(str(hit["content"])) for hit in results)
            return results

        return self._run_logged_tool(
            tool_name="search_kb", model="dense+bm25+rrf", request=request,
            operation=operation, retrieved_tokens=0,
        )

    # ---------- Tool 3: pickup-slot lookup ----------

    def check_pickup_slots(
        self,
        pickup_date: str,
        requested_window: Mapping[str, Any],
        address_summary: str,
    ) -> Sequence[Mapping[str, Any]]:
        request = {
            "pickup_date": str(pickup_date),
            "requested_window": dict(requested_window),
            "address_summary": str(address_summary),
        }

        def operation() -> Sequence[Mapping[str, Any]]:
            if not pickup_date:
                raise ValueError("pickup_date is required")
            requested_start = _time_to_minutes(requested_window.get("start"))
            requested_end = _time_to_minutes(requested_window.get("end"))
            address = str(address_summary or "").casefold()
            matches: list[dict[str, Any]] = []
            with self._exclusive_file_lock(self._runtime_lock_path):
                payload = self._load_runtime_slots_locked()
                for slot in payload["slots"]:
                    if str(slot.get("date")) != str(pickup_date):
                        continue
                    if str(slot.get("status", "")).casefold() != "available":
                        continue
                    postal_area = str(slot.get("postal_area", "")).casefold()
                    if postal_area and postal_area not in address:
                        continue
                    slot_start = _time_to_minutes(str(slot.get("window_start")))
                    slot_end = _time_to_minutes(str(slot.get("window_end")))
                    if requested_start is not None and slot_start is not None and slot_start < requested_start:
                        continue
                    # A requested broad interval may contain one or more actual
                    # slots. If there are several, the dialogue manager asks the
                    # customer to choose instead of guessing.
                    if requested_end is not None and slot_end is not None and slot_end > requested_end:
                        continue
                    matches.append(deepcopy(dict(slot)))
            return matches

        return self._run_logged_tool(
            tool_name="check_pickup_slots", model="json-pickup-database",
            request=request, operation=operation,
        )

    # ---------- Tool 4: confirmed pickup booking database write ----------

    def book_slot(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Book a pickup only after explicit confirmation and live revalidation."""
        safe_args = dict(arguments)
        request = {
            key: safe_args.get(key)
            for key in (
                "customer_id", "order_id", "sku", "reason", "pickup_date",
                "pickup_time_window", "pickup_address_summary", "slot_id",
                "explicit_confirmation",
            )
        }

        def operation() -> Mapping[str, Any]:
            if safe_args.get("explicit_confirmation") is not True:
                raise ConfirmationRequired("Explicit user confirmation is required before booking")
            required = (
                "customer_id", "order_id", "sku", "reason", "pickup_date",
                "pickup_time_window", "pickup_address_summary", "slot_id",
            )
            missing = [key for key in required if not safe_args.get(key)]
            if missing:
                raise ValueError("Missing booking fields: " + ", ".join(missing))

            # Re-check customer ownership and SKU membership immediately before
            # acquiring a slot and writing the booking record.
            order = self._find_order(str(safe_args["order_id"]))
            if str(order.get("customer_id", "")).casefold() != str(safe_args["customer_id"]).casefold():
                raise OrderOwnershipError("The order does not belong to the confirming customer")
            item_skus = {
                str(item.get("sku", "")).casefold()
                for item in (order.get("items") or []) if isinstance(item, Mapping)
            }
            if str(safe_args["sku"]).casefold() not in item_skus:
                raise ValueError("The selected SKU is not part of the requested order")

            requested_window = safe_args["pickup_time_window"]
            requested_start = _time_to_minutes(requested_window.get("start"))
            requested_end = _time_to_minutes(requested_window.get("end"))
            address = str(safe_args["pickup_address_summary"]).casefold()

            with self._exclusive_file_lock(self._runtime_lock_path):
                payload = self._load_runtime_slots_locked()
                slot = next(
                    (row for row in payload["slots"] if str(row.get("slot_id")) == str(safe_args["slot_id"])),
                    None,
                )
                if slot is None:
                    raise SlotUnavailable("The selected pickup slot no longer exists")
                if str(slot.get("status", "")).casefold() != "available":
                    raise SlotUnavailable("The selected pickup slot is no longer available")
                if str(slot.get("date")) != str(safe_args["pickup_date"]):
                    raise SlotUnavailable("The selected slot date does not match the confirmed date")
                postal_area = str(slot.get("postal_area", "")).casefold()
                if postal_area and postal_area not in address:
                    raise SlotUnavailable("The selected slot does not serve the pickup address")

                slot_start = _time_to_minutes(str(slot.get("window_start")))
                slot_end = _time_to_minutes(str(slot.get("window_end")))
                if requested_start is not None and slot_start != requested_start:
                    raise SlotUnavailable("The selected slot start time changed")
                if requested_end is not None and slot_end != requested_end:
                    raise SlotUnavailable("The selected slot end time changed")
                if requested_end is None and requested_start is not None and (slot_start is None or slot_start < requested_start):
                    raise SlotUnavailable("The selected slot is outside the requested time window")

                now = datetime.now().astimezone()
                booking_id = f"PICKUP-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
                rma_id = f"RMA-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
                booking = {
                    "booking_id": booking_id,
                    "rma_id": rma_id,
                    "customer_id": str(safe_args["customer_id"]),
                    "order_id": str(safe_args["order_id"]),
                    "sku": str(safe_args["sku"]),
                    "reason": str(safe_args["reason"]),
                    "pickup_date": str(safe_args["pickup_date"]),
                    "pickup_time_window": {
                        "start": str(slot["window_start"]),
                        "end": str(slot["window_end"]),
                    },
                    "pickup_address_summary": str(safe_args["pickup_address_summary"]),
                    "slot_id": str(slot["slot_id"]),
                    "booking_status": "confirmed",
                    "created_at": now.isoformat(timespec="seconds"),
                    "confirmation_received": True,
                }
                # Mark the slot unavailable and append the real booking record
                # in the same atomic JSON replacement.
                slot["status"] = "booked"
                slot["booking_id"] = booking_id
                payload.setdefault("bookings", []).append(booking)
                self._atomic_write_json(self.runtime_slots_path, payload)
                return {
                    "status": "success",
                    "booking_id": booking_id,
                    "rma_id": rma_id,
                    "booking_status": "confirmed",
                    "slot_id": str(slot["slot_id"]),
                    "order_id": str(safe_args["order_id"]),
                }

        return self._run_logged_tool(
            tool_name="book_slot", model="json-pickup-database",
            request=request, operation=operation,
        )
