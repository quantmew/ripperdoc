"""Tests for context window heuristics."""

from ripperdoc.utils.message_compaction import get_model_context_limit
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
