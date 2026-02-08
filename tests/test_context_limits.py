"""Tests for context window heuristics."""

import pytest

from ripperdoc.utils.message_compaction import (
    ContextBudgetConfigurationError,
    get_model_context_limit,
    get_remaining_context_tokens,
)
from ripperdoc.core.config import ModelProfile, ProtocolType


def test_claude_45_default_and_1m_beta():
    profile_default = ModelProfile(
         protocol=ProtocolType.ANTHROPIC, model="claude-3-5-sonnet-20241022"
    )
    profile_beta = ModelProfile(
         protocol=ProtocolType.ANTHROPIC, model="claude-3-5-sonnet-1m-20241022"
    )

    assert get_model_context_limit(profile_default) == 200_000
    assert get_model_context_limit(profile_beta) == 1_000_000


def test_claude_opus_and_haiku():
    profile_opus = ModelProfile(protocol=ProtocolType.ANTHROPIC, model="claude-4.1")
    profile_haiku = ModelProfile(protocol=ProtocolType.ANTHROPIC, model="claude-3-5-haiku")

    assert get_model_context_limit(profile_opus) >= 200_000
    assert get_model_context_limit(profile_haiku) >= 200_000


def test_openai_and_deepseek_defaults():
    profile_gpt4o = ModelProfile(protocol=ProtocolType.OPENAI_COMPATIBLE, model="gpt-4o")
    profile_gpt4 = ModelProfile(protocol=ProtocolType.OPENAI_COMPATIBLE, model="gpt-4")
    profile_deepseek = ModelProfile(protocol=ProtocolType.OPENAI_COMPATIBLE, model="deepseek-chat")

    assert get_model_context_limit(profile_gpt4o) == 128_000
    assert get_model_context_limit(profile_gpt4) == 32_000
    assert get_model_context_limit(profile_deepseek) >= 128_000


def test_remaining_tokens_uses_split_input_budget_without_double_subtract():
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="zai/glm-4.7",
        max_input_tokens=200_000,
        max_output_tokens=128_000,
        max_tokens=128_000,
    )

    assert get_remaining_context_tokens(profile) == 200_000


def test_remaining_tokens_subtracts_output_for_total_window_profiles():
    profile = ModelProfile(
        protocol=ProtocolType.OPENAI_COMPATIBLE,
        model="example-total-window-model",
        max_tokens=8_192,
    )

    assert get_remaining_context_tokens(profile, explicit_limit=128_000) == 119_808


def test_remaining_tokens_raises_on_output_parse_error():
    class BrokenProfile:
        model = "broken-model"
        max_input_tokens = None
        max_output_tokens = {"invalid": True}
        max_tokens = {"invalid": True}

    with pytest.raises(ContextBudgetConfigurationError):
        get_remaining_context_tokens(BrokenProfile())


def test_remaining_tokens_raises_on_zero_max_input_tokens():
    class BrokenProfile:
        model = "broken-input-model"
        max_input_tokens = 0
        max_output_tokens = 16_384
        max_tokens = 16_384

    with pytest.raises(ContextBudgetConfigurationError):
        get_remaining_context_tokens(BrokenProfile())


def test_remaining_tokens_raises_on_zero_max_output_tokens():
    class BrokenProfile:
        model = "broken-output-model"
        max_input_tokens = None
        max_output_tokens = 0
        max_tokens = 16_384

    with pytest.raises(ContextBudgetConfigurationError):
        get_remaining_context_tokens(BrokenProfile())
