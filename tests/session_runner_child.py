"""Child process for test_run_session: one session with fakes and a temp log.

Usage: python tests/session_runner_child.py <A|B> <tmp_dir> [config]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import src.logger as logger_module  # noqa: E402
from src.dialogue.run_session import run_session  # noqa: E402
from src.memory.fact_extractor import FactExtractor  # noqa: E402
from src.memory.llm_calls import LoggedLLM  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.models import KnowledgeChunk  # noqa: E402
from memory_fakes import FakeEmbedder, FakeOllamaClient  # noqa: E402

session, tmp = sys.argv[1], Path(sys.argv[2])
config = sys.argv[3] if len(sys.argv) > 3 else "hybrid_mem"
logger_module.LOG_PATH = tmp / "calls.jsonl"  # never touch the real evidence log
embedder = FakeEmbedder()


def fake_index_loader(_path: Path) -> DenseIndex:
    chunks = [KnowledgeChunk.from_dict(item) for item in
              json.loads((ROOT / "index" / "dense_128" / "chunks.json").read_text(encoding="utf-8"))]
    return DenseIndex(chunks, embedder.encode([c.content for c in chunks], purpose="test_index"))


extract_llm = LoggedLLM(client=FakeOllamaClient(json.dumps({"facts": [
    {"key": "return_method_preference", "value": "pickup from my home"}]})))
run_session(
    session=session, config=config, experiment=f"t3_test_{config}" if config != "hybrid_mem" else "t3_test",
    use_nlg=False,
    memory_root=tmp / "memory", db_root=tmp / "db", sessions_dir=tmp / "sessions",
    memory_kwargs={
        "embedder": embedder,
        "extractor": FactExtractor(llm=extract_llm),
        "reflection_llm": LoggedLLM(client=FakeOllamaClient(
            f"Session {session} reflection: Maya has a confirmed earbuds return pickup and prefers SMS updates.")),
    },
    tools_kwargs={"index_loader": fake_index_loader},
)
