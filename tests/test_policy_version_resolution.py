from src.dialogue.dialogue_manager import DialogueManager


def test_explicit_active_status_takes_precedence():
    hit = {"doc_id": "KB-900", "status": "active"}
    assert DialogueManager._resolve_policy_status(hit) == "active"


def test_explicit_superseded_status_takes_precedence():
    hit = {"doc_id": "KB-002", "status": "superseded"}
    assert DialogueManager._resolve_policy_status(hit) == "superseded"


def test_manifest_fallback_recognises_kb_002_as_active_when_status_is_missing():
    hit = {"doc_id": "KB-002", "version": "2.0", "chunk_id": "KB-002-C128-0001"}
    assert DialogueManager._resolve_policy_status(hit) == "active"


def test_manifest_fallback_recognises_kb_001_as_superseded_when_status_is_missing():
    hit = {"document_id": "KB-001", "version": "1.0"}
    assert DialogueManager._resolve_policy_status(hit) == "superseded"


def test_current_heading_can_supply_missing_status():
    hit = {"doc_id": "KB-900", "section": "Returns Policy — 2026 (Current)"}
    assert DialogueManager._resolve_policy_status(hit) == "active"


def test_superseded_heading_can_supply_missing_status():
    hit = {"doc_id": "KB-900", "section": "Returns Policy — 2025 (Superseded)"}
    assert DialogueManager._resolve_policy_status(hit) == "superseded"


def test_unknown_document_without_status_fails_closed():
    hit = {"doc_id": "KB-999", "version": "9.0", "section": "Other policy"}
    assert DialogueManager._resolve_policy_status(hit) == "unknown"


def test_explicit_unknown_status_is_not_overridden_by_manifest_fallback():
    hit = {"doc_id": "KB-002", "status": "pending_review"}
    assert DialogueManager._resolve_policy_status(hit) == "pending_review"
