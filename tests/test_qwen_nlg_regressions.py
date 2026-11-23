import json

import pytest

from src.dialogue.qwen_nlg import QwenNLG


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "message": {"content": "Hello! I can help with ShopSphere."},
            "prompt_eval_count": 12,
            "eval_count": 8,
        }
        self.error = error
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _make_nlg(tmp_path, client, logged):
    prompt = tmp_path / "system_prompt_v1.txt"
    prompt.write_text("You are RecallBot. Keep replies concise.", encoding="utf-8")
    workflow = tmp_path / "return_pickup_workflow_v1.json"
    workflow.write_text(json.dumps({"version": "1.0", "required_slots": [], "steps": []}), encoding="utf-8")
    return QwenNLG(
        system_prompt_path=prompt,
        workflow_path=workflow,
        client=client,
        logger=lambda **kwargs: logged.append(kwargs),
        token_counter=lambda text: len(text.split()),
        context_window=2048,
    )


def test_qwen_chat_explicitly_disables_thinking_for_customer_facing_nlg(tmp_path):
    logged = []
    client = FakeClient()
    nlg = _make_nlg(tmp_path, client, logged)

    reply = nlg.generate_reply(
        "Hello! I can help with ShopSphere.",
        user_message="Hi",
        intent="greeting",
        state={},
        turn_type="open",
    )

    assert reply == "Hello! I can help with ShopSphere."
    assert client.kwargs["think"] is False
    assert logged[-1]["call_type"] == "chat"
    assert logged[-1]["extra"]["status"] == "success"


def test_qwen_failure_message_is_logged_without_masking_exception(tmp_path):
    logged = []
    client = FakeClient(error=RuntimeError("diagnostic test failure"))
    nlg = _make_nlg(tmp_path, client, logged)

    with pytest.raises(RuntimeError, match="diagnostic test failure"):
        nlg.generate_reply(
            "Hello.",
            user_message="Hi",
            intent="greeting",
            state={},
            turn_type="open",
        )

    assert logged[-1]["extra"]["status"] == "failed"
    assert logged[-1]["extra"]["error_type"] == "RuntimeError"
    assert logged[-1]["extra"]["error_message"] == "diagnostic test failure"
