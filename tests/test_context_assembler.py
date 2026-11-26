from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context.context_assembler import FALLBACK_SENTENCE, ContextAssembler, build_turn_items
from src.dialogue.dialogue_manager import DialogueManager
from src.dialogue.nlu import analyze_utterance
from src.dialogue.state_tracker import DialogueStateTracker
from src.memory.episodic_memory import ScoredEpisode
from src.memory.fact_extractor import FactExtractor
from src.memory.memory_manager import MemoryManager
from src.memory.records import Episode, SemanticFact
from memory_fakes import FakeEmbedder, RecordingLogger

ROOT = Path(__file__).resolve().parents[1]
USER = "CUST-MAYA-1042"
WORDS = lambda text: len(text.split())  # noqa: E731 - deterministic test counter


def _fact(memory_id="SM-0004", key="preferred_delivery_window", value="19:00-21:00"):
    return SemanticFact(memory_id, USER, key, value, 19, "A", "2026-11-21T10:36:00+05:30")


def _episode(episode_id="EP-0009", description="Return pickup booked for SKU-AUR-220, RMA RMA-1."):
    ep = Episode(episode_id, USER, "A", "2026-11-21T10:44:00+05:30", 23, "booking", description, 9)
    return ScoredEpisode(ep, 0.7, 0.9, 0.8, 1.0, 1.0, 1.0, 1.0)


def _items(intent, *, kb=2, history_turns=3, filler=40):
    chunk = " ".join(["policy"] * filler)
    history = []
    for turn in range(1, history_turns + 1):
        history += [{"role": "user", "content": f"user turn {turn} " + "w " * filler},
                    {"role": "assistant", "content": f"reply {turn} " + "w " * filler}]
    return build_turn_items(
        intent=intent, system_prompt="SYSTEM " * 30, system_prompt_version="system_prompt_v2",
        workflow_step=None, user_message="What delivery window do I prefer?", state={"turn": 27},
        draft="DRAFT", facts=[(_fact(), 0.8)], episodes=[_episode()],
        kb_hits=[{"chunk_id": f"KB-002-C128-000{i}", "doc_id": "KB-002", "version": "2026", "status": "active",
                  "effective_from": "2026-01-01", "content": chunk} for i in range(kb)],
        history=history,
    )


def test_everything_fits_when_the_budget_allows():
    items, profile = _items("semantic_memory_query")
    assembled = ContextAssembler(token_counter=WORDS).assemble(items, output_reserve=160, profile=profile)
    assert assembled.dropped == []
    assert assembled.input_budget == 2048 - 160 - 32 == 1856
    assert assembled.total_input_tokens <= assembled.input_budget
    assert [m["role"] for m in assembled.messages] == ["system", "user", "assistant", "user", "assistant",
                                                       "user", "assistant", "user"]
    assert "[SM-0004]" in assembled.messages[-1]["content"] and "[EP-0009]" in assembled.messages[-1]["content"]


def test_memory_profile_keeps_memory_and_drops_history_then_kb_when_budget_is_tight():
    items, profile = _items("semantic_memory_query", filler=120)
    assembler = ContextAssembler(token_counter=WORDS, context_window=900)
    assembled = assembler.assemble(items, output_reserve=160, profile=profile)
    included = {item["id"] for item in assembled.included}
    assert {"SM-0004", "EP-0009"} <= included
    dropped = [item["id"] for item in assembled.dropped]
    assert dropped and all(item["reason"].startswith("budget") for item in assembled.dropped)
    # History is dropped oldest first; the newest exchange survives the longest.
    assert "HISTORY-1" in dropped and dropped.index("HISTORY-1") < dropped.index("HISTORY-3") \
        if "HISTORY-3" in dropped else "HISTORY-1" in dropped
    assert assembled.total_input_tokens <= assembled.input_budget
    assert assembled.retrieved_tokens == sum(
        i["tokens"] for i in assembled.included if i["component"] in {"long_term_memory", "retrieved_knowledge"})


def test_knowledge_profile_ranks_kb_chunks_before_memories_and_drops_whole_chunks():
    items, profile = _items("knowledge_base_query", kb=3, filler=150, history_turns=0)
    assert profile == "knowledge"
    priority = {item.item_id: item.priority for item in items if not item.mandatory}
    assert max(priority[f"KB-002-C128-000{i}"] for i in range(3)) < min(priority["SM-0004"], priority["EP-0009"])

    # Budget 600-160-32 = 408: two 156-token chunks fit after the mandatory items;
    # the third chunk and both lower-priority memories are cut and recorded.
    assembled = ContextAssembler(token_counter=WORDS, context_window=600).assemble(
        items, output_reserve=160, profile=profile)
    included = [item["id"] for item in assembled.included]
    dropped = [item["id"] for item in assembled.dropped]
    assert {"KB-002-C128-0000", "KB-002-C128-0001"} <= set(included)
    assert dropped == ["KB-002-C128-0002", "SM-0004", "EP-0009"]
    assert assembled.total_input_tokens <= 408


def test_mandatory_context_that_cannot_fit_raises_for_deterministic_fallback():
    items, profile = _items("knowledge_base_query")
    with pytest.raises(ValueError, match="Mandatory context"):
        ContextAssembler(token_counter=WORDS, context_window=200).assemble(items, output_reserve=160, profile=profile)


def test_real_qwen_template_count_for_a_full_session_b_turn_stays_within_budget():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer is not cached locally")
    chunks = json.loads((ROOT / "index" / "dense_128" / "chunks.json").read_text(encoding="utf-8"))[:3]
    assembler = ContextAssembler(tokenizer=tokenizer)
    history = [{"role": r, "content": "A fairly ordinary customer-support message about an order. " * 6}
               for _ in range(3) for r in ("user", "assistant")]
    items, profile = build_turn_items(
        intent="knowledge_base_query",
        system_prompt=(ROOT / "skills" / "system_prompt_v2.txt").read_text(encoding="utf-8"),
        system_prompt_version="system_prompt_v2", workflow_step=None,
        user_message="For a 2026 delivery, how many days do I have to return electronic accessories?",
        state={"session_id": "B", "turn": 31, "active_task": None, "task_status": "idle"},
        draft="Answer from the KB chunks.", facts=[(_fact(), 0.7)], episodes=[_episode()],
        kb_hits=[{**c, "doc_id": c["doc_id"]} for c in chunks], history=history, truncate=assembler.truncate,
    )
    assembled = assembler.assemble(items, output_reserve=160, profile=profile)
    assert assembled.token_count_method == "qwen3_chat_template+ollama_offset"
    # Plausibility: the templated prompt must be at least the sum of its parts
    # (a BatchEncoding miscount once reported 2 tokens for every prompt).
    content_tokens = sum(item["tokens"] for item in assembled.included)
    assert content_tokens > 300
    assert content_tokens <= assembled.total_input_tokens <= content_tokens + 120
    assert assembled.total_input_tokens <= 1856
    table = assembled.component_table()
    assert sum(row["tokens"] for row in table) == assembled.total_input_tokens
    assert all(row["tokens"] >= 0 for row in table)


# ----------------------------------------------------------- DM memory reads
class NoTools:
    def search_kb(self, query, top_k=3):
        raise AssertionError("no_rag must not call search_kb")


class CapturingNLG:
    def __init__(self, reply="NLG reply"):
        self.calls, self.reply = [], reply

    def set_call_context(self, **kwargs):
        pass

    def generate_reply(self, draft, **kwargs):
        self.calls.append({"draft": draft, **kwargs})
        return self.reply


def _manager(tmp_path, config="hybrid_mem", nlg=None):
    memory = MemoryManager(tmp_path, config=config, user=USER, embedder=FakeEmbedder(), logger=None,
                           extractor=FactExtractor(enable_llm_fallback=False, logger=None, token_counter=WORDS),
                           token_counter=WORDS)
    tracker = DialogueStateTracker(session_id="B", user_id=USER)
    dm = DialogueManager(tracker=tracker, tools=NoTools(), user_id=USER, config=config, logger=None,
                         memory=memory, nlg=nlg, token_counter=WORDS)
    dm.context_assembler = ContextAssembler(token_counter=WORDS)
    return dm, memory


def _seed(memory):
    memory.set_call_context(user=USER, session="A", turn=19)
    text = "My preferred delivery window has changed to 7 pm to 9 pm."
    memory.observe_user_turn(text, analyze_utterance(text))
    memory.set_call_context(user=USER, session="A", turn=6)
    text = "If I am not home, leave small parcels at the side door when the carrier permits it."
    memory.observe_user_turn(text, analyze_utterance(text))
    memory.set_call_context(user=USER, session="A", turn=24)
    text = "Please forget my saved delivery note about leaving parcels at the side door."
    memory.forget(text, analyze_utterance(text))


def test_named_fact_question_answers_from_that_key_with_its_id(tmp_path):
    dm, memory = _manager(tmp_path)
    _seed(memory)
    result = dm.handle_turn("Which delivery window did I most recently ask you to save?", turn=27)
    fact = memory.semantic.get_active(USER, "preferred_delivery_window")
    assert "19:00-21:00" in result["reply"] and f"[{fact.memory_id}]" in result["reply"]


def test_unknown_personal_fact_returns_the_exact_fallback_without_the_model(tmp_path):
    nlg = CapturingNLG()
    dm, memory = _manager(tmp_path, nlg=nlg)
    _seed(memory)
    result = dm.handle_turn("What is my pet's name?", turn=30)
    assert result["reply"] == FALLBACK_SENTENCE
    assert result["nlg_status"] == "verbatim" and nlg.calls == []


def test_forgotten_question_never_reveals_value_and_never_triggers_a_new_forget(tmp_path):
    dm, memory = _manager(tmp_path, nlg=CapturingNLG())
    _seed(memory)
    tombstones_before = len(memory.tombstones)
    result = dm.handle_turn("What delivery note did I ask you to forget?", turn=30)
    assert result["intent"] == "forgotten_memory_query"
    assert "deleted at your request" in result["reply"] and "side door" not in result["reply"]
    assert len(memory.tombstones) == tombstones_before


def test_model_output_containing_a_forgotten_value_is_replaced(tmp_path):
    dm, memory = _manager(tmp_path, nlg=CapturingNLG(reply="Sure! I will leave it at the side door."))
    _seed(memory)
    result = dm.handle_turn("Which delivery window did I most recently ask you to save?", turn=27)
    assert "side door" not in result["reply"] and result["nlg_status"].endswith("+forgotten_guard")


def test_assembled_prompt_carries_memory_ids_and_logs_retrieved_tokens(tmp_path):
    nlg = CapturingNLG()
    dm, memory = _manager(tmp_path, nlg=nlg)
    _seed(memory)
    result = dm.handle_turn("Which delivery window did I most recently ask you to save?", turn=27)
    call = nlg.calls[0]
    fact_id = memory.semantic.get_active(USER, "preferred_delivery_window").memory_id
    assert f"[{fact_id}]" in call["prebuilt_messages"][-1]["content"]
    assert call["context_assembly"]["prompt_version"] == "system_prompt_v2"
    assert call["prebuilt_retrieved_tokens"] == result["context_assembly"]["retrieved_tokens"] > 0
    assert all(m["content"].count("side door") == 0 for m in call["prebuilt_messages"])


def test_trim_cuts_are_recorded_and_budget_answer_is_verbatim_with_live_numbers(tmp_path):
    nlg = CapturingNLG()
    dm, memory = _manager(tmp_path, nlg=nlg)
    for turn in range(1, 6):
        result = dm.handle_turn("Thanks, that's helpful.", turn=turn)
    # Turn 5 sees 4 completed exchanges; trim keeps 3, so exchange 1 is cut and recorded.
    assert [item["id"] for item in result["context_assembly"]["dropped_by_trim"]] == ["EXCHANGE-1"]
    previous = result["context_assembly"]

    answer = dm.handle_turn("What is the exact token limit your assistant is configured to respect?", turn=6)
    assert answer["nlg_status"] == "verbatim"
    assert "1,856 input tokens" in answer["reply"]
    # The previous turn was small talk: an open turn reserves 128 output tokens (budget 1,888).
    assert previous["input_budget"] == 1888
    assert f"used {previous['total_input_tokens']:,} of 1,888" in answer["reply"]
    assert "1 item(s) were cut" in answer["reply"]
    assert len(nlg.calls) == 5  # the budget answer never reached the model


def test_no_rag_never_calls_search_kb_and_falls_back_without_a_model(tmp_path):
    dm, _ = _manager(tmp_path, config="no_rag")
    result = dm.handle_turn("What does code ERR-RMA-04 mean?", turn=18)
    assert result["reply"] == FALLBACK_SENTENCE
