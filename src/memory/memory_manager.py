"""Long-term memory facade: writes, retrieval, forget cascade, persistence.

One MemoryManager owns one memory directory (``memory/runs/<experiment>/``).
Session A and Session B are separate processes; everything Session B knows
comes from the files written here, never from Python objects.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..logger import log_call, timer
from .clock import SessionClock
from .episodic_memory import (
    DEFAULT_WEIGHTING, WEIGHTINGS, EpisodicMemoryStream, EpisodicWeights, ScoredEpisode, rule_importance,
)
from .fact_extractor import ExtractionResult, FactExtractor
from .forgetting import contains_fingerprint, fingerprint, placeholder_for, redact, resolve_forget_keys
from .llm_calls import LoggedLLM, default_token_counter
from .records import Episode, SemanticFact, key_label
from .reflection import generate_reflection, reflection_inputs
from .semantic_memory import SemanticMemoryStore
from .vector_store import MemoryVectorIndex

# Which long-term memory types each configuration may write and read.
CONFIG_MEMORY: dict[str, tuple[bool, bool]] = {
    "full": (False, False), "trim": (False, False), "summary": (False, False),
    "no_rag": (False, False), "hybrid": (False, False),
    "hybrid_sem": (True, False), "hybrid_mem": (True, True),
}
MEMORY_FILES = (
    "semantic_facts.json", "semantic_vectors.npy", "semantic_index.json",
    "episodes.json", "episodic_vectors.npy", "episodic_index.json",
    "forget_tombstones.json",
)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Minimum raw BGE cosine similarity for a memory to enter the prompt,
# calibrated once on the pilot Session A memory and then fixed
# (evidence: results/task3_threshold_calibration.json). No single fact
# threshold separated relevant from irrelevant queries (min relevant 0.663,
# max irrelevant 0.704), so the threshold depends on the NLU intent:
# an explicit memory question uses the lower value; every other turn uses a
# value above all calibration negatives so memories do not leak into tasks.
FACT_MIN_SCORE_QUERY = 0.60
FACT_MIN_SCORE_CONTEXT = 0.72
EPISODE_MIN_RELEVANCE_QUERY = 0.50
EPISODE_MIN_RELEVANCE_CONTEXT = 0.80
FACT_MIN_SCORE = FACT_MIN_SCORE_CONTEXT
EPISODE_MIN_RELEVANCE = EPISODE_MIN_RELEVANCE_CONTEXT


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MemoryManager:
    def __init__(
        self,
        memory_dir: Path,
        *,
        config: str,
        user: str,
        embedder: Any | None = None,
        embedder_factory: Callable[[str], Any] | None = None,
        extractor: FactExtractor | None = None,
        reflection_llm: LoggedLLM | None = None,
        logger: Callable[..., Any] | None = log_call,
        token_counter: Callable[[str], int] = default_token_counter,
        weighting: str = DEFAULT_WEIGHTING,
        autosave: bool = True,
    ) -> None:
        if config not in CONFIG_MEMORY:
            raise ValueError(f"Unknown config {config!r}")
        self.memory_dir = Path(memory_dir)
        self.config = config
        self.semantic_enabled, self.episodic_enabled = CONFIG_MEMORY[config]
        self.user = user
        self._embedder = embedder
        self._embedder_factory = embedder_factory
        self.logger = logger
        self.token_counter = token_counter
        self.extractor = extractor if extractor is not None else FactExtractor(logger=logger, token_counter=token_counter)
        self.reflection_llm = reflection_llm
        self.weights: EpisodicWeights = WEIGHTINGS[weighting]
        self.autosave = autosave
        self.semantic = SemanticMemoryStore()
        self.episodic = EpisodicMemoryStream()
        self.tombstones: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {"user": user, "session": "A", "turn": 0, "config": config}
        self._load()

    # ------------------------------------------------------------------ context
    def set_call_context(self, *, user: str, session: str, turn: int, config: str | None = None) -> None:
        self._context = {"user": user, "session": session, "turn": int(turn), "config": config or self.config}

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def now(self, turn: int | None = None) -> str:
        return SessionClock.for_session(self._context["session"]).iso_at(
            self._context["turn"] if turn is None else turn
        )

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            if self._embedder_factory is not None:
                self._embedder = self._embedder_factory(EMBEDDING_MODEL)
            else:
                from ..retrieval.embeddings import SentenceTransformerEmbedder
                self._embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)
        return self._embedder

    def _embed(self, texts: list[str], *, is_query: bool, purpose: str) -> np.ndarray:
        ctx = self._context
        return self._get_embedder().encode(
            texts, purpose=purpose, is_query=is_query, user=ctx["user"], session=ctx["session"],
            turn=ctx["turn"], config=ctx["config"], show_progress_bar=False,
        )

    # ------------------------------------------------------------ semantic write
    def observe_user_turn(self, text: str, nlu: Mapping[str, Any]) -> dict[str, Any]:
        """Extract facts from one user turn and store/supersede them."""
        outcome: dict[str, Any] = {"extraction": None, "actions": []}
        if not self.semantic_enabled:
            return outcome
        result: ExtractionResult = self.extractor.extract(text, nlu, context=self._context)
        outcome["extraction"] = {"status": result.status, "fallback_used": result.fallback_used,
                                 "rejected": result.rejected}
        for fact in result.facts:
            if not self.semantic.would_change(self.user, fact.key, fact.value):
                current = self.semantic.get_active(self.user, fact.key)
                outcome["actions"].append({"action": "unchanged", "record": current.to_dict() if current else None})
                continue
            probe = SemanticFact("SM-PENDING", self.user, fact.key, fact.value, 0, "", "")
            vector = self._embed([probe.embedding_text()], is_query=False, purpose="semantic_memory_write")[0]
            action, record, superseded = self.semantic.upsert(
                user=self.user, key=fact.key, value=fact.value, source_turn=self._context["turn"],
                session=self._context["session"], date=self.now(), vector=vector,
                extraction_method=fact.method, rule_id=fact.rule_id,
            )
            outcome["actions"].append({
                "action": action, "record": record.to_dict(),
                "superseded": superseded.to_dict() if superseded else None,
            })
            if self.episodic_enabled:
                if superseded is None:
                    description = f"Customer saved their {key_label(record.key)}: {record.value}."
                    episode_type = "preference_event"
                else:
                    description = (f"Customer changed their {key_label(record.key)} from "
                                   f"{superseded.value} to {record.value}.")
                    episode_type = "preference_update"
                linked = [record.memory_id] + ([superseded.memory_id] if superseded else [])
                self.record_episode(episode_type, description, linked_memory_ids=linked,
                                    entities={"key": record.key})
        if outcome["actions"]:
            self._autosave()
        return outcome

    # ------------------------------------------------------------ episodic write
    def record_episode(
        self,
        episode_type: str,
        description: str,
        *,
        entities: Mapping[str, Any] | None = None,
        linked_memory_ids: Iterable[str] | None = None,
        timestamp: str | None = None,
        importance: int | None = None,
        importance_method: str = "rule",
    ) -> Episode | None:
        if not self.episodic_enabled:
            return None
        description = self._redact_forgotten(description)
        started = timer()
        score = rule_importance(episode_type) if importance is None else int(importance)
        if self.logger is not None and importance_method == "rule":
            self.logger(
                config=self._context["config"], user=self._context["user"],
                session=self._context["session"], turn=self._context["turn"],
                call_type="importance", model="rule-based",
                input_tokens=self.token_counter(description), output_tokens=0, retrieved_tokens=0,
                start_time=started,
                extra={"purpose": "episode_importance", "episode_type": episode_type, "importance": score},
            )
        probe_text = f"{episode_type.replace('_', ' ')}: {description}"
        vector = self._embed([probe_text], is_query=False, purpose="episodic_memory_write")[0]
        episode = self.episodic.add(
            user=self.user, session=self._context["session"], timestamp=timestamp or self.now(),
            source_turn=self._context["turn"], episode_type=episode_type, description=description,
            vector=vector, importance=score, importance_method=importance_method,
            entities=dict(entities or {}), linked_memory_ids=list(linked_memory_ids or []),
        )
        if episode is not None:
            self._autosave()
        return episode

    # ------------------------------------------------------------------ forget
    def forget(self, text: str, nlu: Mapping[str, Any]) -> dict[str, Any]:
        """Delete a fact everywhere: records, index, episodes and reflections."""
        keys = resolve_forget_keys(text, nlu)
        report: dict[str, Any] = {"keys": keys, "facts_removed": [], "episodes_deleted": [],
                                  "episodes_redacted": []}
        if not keys:
            return report
        entities = nlu.get("entities") or {}
        for key in keys:
            removed = self.semantic.forget_key(self.user, key)
            surface_forms = {fact.value for fact in removed}
            for entity, mapped in (("delivery_note", "delivery_note"), ("notification_channel", "notification_channel"),
                                   ("city", "preferred_city"), ("preferred_name", "preferred_name"),
                                   ("packaging_preference", "packaging_preference")):
                if mapped == key and isinstance(entities.get(entity), str):
                    surface_forms.add(entities[entity])
            fingerprints = [fingerprint(value) for value in sorted(surface_forms) if value.strip()]
            report["facts_removed"].extend(fact.memory_id for fact in removed)
            removed_ids = {fact.memory_id for fact in removed}

            linked = [
                ep.episode_id for ep in self.episodic.episodes(self.user)
                if ep.type in {"preference_event", "preference_update"} and removed_ids & set(ep.linked_memory_ids)
            ]
            report["episodes_deleted"].extend(ep.episode_id for ep in self.episodic.delete(linked))

            label = placeholder_for(key)
            for episode in self.episodic.episodes(self.user):
                if fingerprints and contains_fingerprint(episode.description, fingerprints):
                    new_text = redact(episode.description, fingerprints, label)
                    vector = self._embed([f"{episode.type.replace('_', ' ')}: {new_text}"],
                                         is_query=False, purpose="episodic_memory_redaction")[0]
                    self.episodic.replace_text(episode.episode_id, new_text, vector)
                    report["episodes_redacted"].append(episode.episode_id)

            if fingerprints:
                self.tombstones.append({
                    "user": self.user, "key": key, "fingerprints": fingerprints,
                    "forgotten_at": self.now(), "source_turn": self._context["turn"],
                    "session": self._context["session"],
                })
        if self.episodic_enabled and report["facts_removed"]:
            labels = ", ".join(key_label(key) for key in keys)
            self.record_episode("forget_event", f"Customer asked RecallBot to forget their saved {labels}.",
                                entities={"keys": keys})
        self._autosave()
        return report

    def _fingerprints(self) -> list[dict[str, Any]]:
        return [fp for stone in self.tombstones if stone["user"] == self.user for fp in stone["fingerprints"]]

    def _redact_forgotten(self, text: str) -> str:
        for stone in self.tombstones:
            if stone["user"] == self.user:
                text = redact(text, stone["fingerprints"], placeholder_for(stone["key"]))
        return text

    def redact_text(self, text: str) -> str:
        """Public guard for prompts, history and generated text."""
        return self._redact_forgotten(text)

    def contains_forgotten(self, text: str) -> bool:
        return contains_fingerprint(text, self._fingerprints())

    # --------------------------------------------------------------- retrieval
    def retrieve_for_turn(
        self,
        query: str,
        *,
        fact_top_k: int = 2,
        episode_top_k: int = 2,
        fact_min_score: float | None = None,
        episode_min_relevance: float | None = None,
    ) -> dict[str, list[Any]]:
        """One query embedding, reused for both fact and episode retrieval."""
        result: dict[str, list[Any]] = {"facts": [], "episodes": []}
        want_facts = self.semantic_enabled and bool(self.semantic.active_facts(self.user))
        want_episodes = self.episodic_enabled and bool(self.episodic.episodes(self.user))
        if not (want_facts or want_episodes):
            return result
        vector = self._embed([query], is_query=True, purpose="memory_search")[0]
        if want_facts:
            result["facts"] = self.semantic.search(
                self.user, vector, top_k=fact_top_k,
                min_score=FACT_MIN_SCORE if fact_min_score is None else fact_min_score)
        if want_episodes:
            result["episodes"] = self.episodic.retrieve(
                self.user, vector, now=self.now(), weights=self.weights, top_k=episode_top_k,
                min_relevance=EPISODE_MIN_RELEVANCE if episode_min_relevance is None else episode_min_relevance)
        return result

    def tombstone_for(self, key: str) -> dict[str, Any] | None:
        return next((s for s in self.tombstones if s["user"] == self.user and s["key"] == key), None)

    def retrieve_facts(self, query: str, *, top_k: int = 3, min_score: float = 0.0) -> list[tuple[SemanticFact, float]]:
        if not self.semantic_enabled or not self.semantic.active_facts(self.user):
            return []
        vector = self._embed([query], is_query=True, purpose="semantic_memory_search")[0]
        return self.semantic.search(self.user, vector, top_k=top_k, min_score=min_score)

    def retrieve_episodes(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_relevance: float = 0.0,
        weights: EpisodicWeights | None = None,
        types: Iterable[str] | None = None,
        now: str | None = None,
    ) -> list[ScoredEpisode]:
        if not self.episodic_enabled or not self.episodic.episodes(self.user):
            return []
        vector = self._embed([query], is_query=True, purpose="episodic_memory_search")[0]
        return self.episodic.retrieve(
            self.user, vector, now=now or self.now(), weights=weights or self.weights,
            top_k=top_k, min_relevance=min_relevance, types=types,
        )

    # -------------------------------------------------------------- reflection
    def write_reflection(self, *, session: str, last_turn: int) -> Episode | None:
        """Reflect on this session's notable episodes and store the insight."""
        if not self.episodic_enabled:
            return None
        session_episodes = [ep for ep in self.episodic.episodes(self.user) if ep.session == session]
        selected, payload = reflection_inputs(session_episodes, self.semantic.active_facts(self.user))
        if not selected:
            return None
        if self.reflection_llm is None:
            self.reflection_llm = LoggedLLM(logger=self.logger, token_counter=self.token_counter)
        self.set_call_context(user=self.user, session=session, turn=last_turn, config=self.config)
        text = generate_reflection(self.reflection_llm, payload, context=self._context)
        return self.record_episode(
            "reflection", text, linked_memory_ids=[ep.episode_id for ep in selected],
            entities={"session": session}, timestamp=self.now(last_turn + 1),
        )

    # -------------------------------------------------------------- persistence
    def save(self) -> dict[str, str]:
        directory = self.memory_dir
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(directory / "semantic_facts.json", _json_bytes(self.semantic.to_dict()))
        _atomic_write_bytes(directory / "episodes.json", _json_bytes(self.episodic.to_dict()))
        _atomic_write_bytes(directory / "forget_tombstones.json", _json_bytes({"tombstones": self.tombstones}))
        for name, index in (("semantic", self.semantic.index), ("episodic", self.episodic.index)):
            index.save(directory, name)
        return self.file_hashes()

    def file_hashes(self) -> dict[str, str]:
        return {name: file_sha256(self.memory_dir / name) for name in MEMORY_FILES if (self.memory_dir / name).exists()}

    def _autosave(self) -> None:
        if self.autosave:
            self.save()

    def _load(self) -> None:
        directory = self.memory_dir
        facts_path = directory / "semantic_facts.json"
        episodes_path = directory / "episodes.json"
        tombstones_path = directory / "forget_tombstones.json"
        if facts_path.exists():
            self.semantic = SemanticMemoryStore.from_dict(
                json.loads(facts_path.read_text(encoding="utf-8")), MemoryVectorIndex.load(directory, "semantic"))
        if episodes_path.exists():
            self.episodic = EpisodicMemoryStream.from_dict(
                json.loads(episodes_path.read_text(encoding="utf-8")), MemoryVectorIndex.load(directory, "episodic"))
        if tombstones_path.exists():
            self.tombstones = json.loads(tombstones_path.read_text(encoding="utf-8")).get("tombstones", [])
