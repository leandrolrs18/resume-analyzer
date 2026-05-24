import sys
from types import SimpleNamespace

from app.services.llm_service import LlmService


def test_llm_service_loads_gguf_model_and_generates_with_chatml(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake-gguf")

    class DummyLlama:
        def __init__(self, **kwargs):
            captured["load_kwargs"] = kwargs

        def __call__(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["generate_kwargs"] = kwargs
            return {"choices": [{"text": '{"candidates": []}'}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=DummyLlama))

    service = LlmService(str(model_path))
    output = service._generate_sync("Use somente evidências.", max_new_tokens=80)

    assert output == '{"candidates": []}'
    assert captured["load_kwargs"] == {
        "model_path": str(model_path),
        "n_ctx": 2048,
        "n_threads": 4,
        "verbose": False,
    }
    assert "<|im_start|>system" in str(captured["prompt"])
    assert "Use somente evidências." in str(captured["prompt"])
    assert captured["generate_kwargs"] == {
        "max_tokens": 80,
        "temperature": 0.1,
        "stop": ["<|im_end|>", "<|im_start|>"],
    }
