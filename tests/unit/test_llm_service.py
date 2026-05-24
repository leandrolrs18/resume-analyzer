from types import SimpleNamespace

from app.services.llm_service import LlmService


def test_llm_service_loads_model_without_accelerate_only_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}
    torch_stub = SimpleNamespace(float32="float32")

    class DummyTokenizer:
        pass

    class DummyModel:
        pass

    class DummyConfig:
        is_encoder_decoder = True

    def fake_config_from_pretrained(model_name: str):
        captured["config_model_name"] = model_name
        return DummyConfig()

    def fake_tokenizer_from_pretrained(model_name: str):
        captured["tokenizer_model_name"] = model_name
        return DummyTokenizer()

    def fake_model_from_pretrained(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["model_kwargs"] = kwargs
        return DummyModel()

    def fake_pipeline(task: str, model: object, tokenizer: object, device: int):
        captured["pipeline_task"] = task
        captured["pipeline_model"] = model
        captured["pipeline_tokenizer"] = tokenizer
        captured["pipeline_device"] = device
        return object()

    class DummyAutoConfig:
        from_pretrained = staticmethod(fake_config_from_pretrained)

    class DummyAutoCausalModel:
        from_pretrained = staticmethod(fake_model_from_pretrained)

    class DummyAutoSeq2SeqModel:
        from_pretrained = staticmethod(fake_model_from_pretrained)

    class DummyAutoTokenizer:
        from_pretrained = staticmethod(fake_tokenizer_from_pretrained)

    monkeypatch.setattr(
        "app.services.llm_service.load_llm_dependencies",
        lambda: (
            torch_stub,
            DummyAutoConfig,
            DummyAutoCausalModel,
            DummyAutoSeq2SeqModel,
            DummyAutoTokenizer,
            fake_pipeline,
        ),
    )

    service = LlmService("test-model")
    service._load_generator()

    assert captured["config_model_name"] == "test-model"
    assert captured["tokenizer_model_name"] == "test-model"
    assert captured["model_name"] == "test-model"
    assert captured["model_kwargs"] == {
        "torch_dtype": "float32",
        "low_cpu_mem_usage": False,
    }
    assert captured["pipeline_task"] == "text2text-generation"
    assert captured["pipeline_device"] == -1
