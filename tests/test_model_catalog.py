"""Tests for packaged model catalog metadata lookup and defaults."""

from ripperdoc.core.config import ModelProfile, ProtocolType
from ripperdoc.core.model_catalog import get_catalog_size, lookup_model_metadata


def test_catalog_is_packaged_and_non_empty():
    assert get_catalog_size() > 2_000


def test_lookup_exact_model_metadata():
    metadata = lookup_model_metadata("gpt-4o-mini", ProtocolType.OPENAI_COMPATIBLE)
    assert metadata is not None
    assert metadata.key == "gpt-4o-mini"
    assert metadata.max_input_tokens is not None
    assert metadata.max_output_tokens is not None


def test_lookup_fuzzy_model_metadata():
    metadata = lookup_model_metadata("glm-4.7", ProtocolType.OPENAI_COMPATIBLE)
    assert metadata is not None
    assert "glm-4.7" in metadata.key


def test_model_profile_uses_catalog_defaults():
    profile = ModelProfile(protocol=ProtocolType.OPENAI_COMPATIBLE, model="gpt-4o-mini")

    assert profile.max_input_tokens is not None and profile.max_input_tokens >= 128_000
    assert profile.max_output_tokens is not None and profile.max_output_tokens >= 4_096
    assert profile.max_tokens <= profile.max_output_tokens
    assert profile.max_tokens <= 8_192
    assert profile.mode in {"chat", "completion", "responses"}
    assert profile.supports_vision is True
    assert profile.price.input > 0
    assert profile.price.output > 0
    assert profile.currency == "USD"


def test_model_profile_reasoning_flag_is_inferred():
    profile = ModelProfile(protocol=ProtocolType.OPENAI_COMPATIBLE, model="deepseek-reasoner")
    assert profile.supports_reasoning is True


def test_explicit_values_are_not_overridden():
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="gpt-4o-mini",
        max_tokens=2_048,
        supports_vision=False,
        price={"input": 9.0, "output": 10.0},
        currency="CNY",
    )
    assert profile.max_tokens == 2_048
    assert profile.supports_vision is False
    assert profile.price.input == 9.0
    assert profile.price.output == 10.0
    assert profile.currency == "CNY"
