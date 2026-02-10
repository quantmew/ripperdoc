"""Tests for session usage aggregation helpers."""

import pytest

from ripperdoc.utils.messages import create_assistant_message, create_user_message
from ripperdoc.utils.session_usage import (
    get_session_usage,
    rebuild_session_usage,
    reset_session_usage,
)


def test_rebuild_session_usage_from_assistant_messages() -> None:
    reset_session_usage()
    messages = [
        create_user_message("hello"),
        create_assistant_message(
            "a1",
            model="gpt-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_creation_tokens=10,
            duration_ms=1200,
            cost_usd=0.0123,
        ),
        create_assistant_message(
            "a2",
            model="gpt-5",
            input_tokens=30,
            output_tokens=40,
            duration_ms=300,
            cost_usd=0.004,
        ),
    ]

    usage = rebuild_session_usage(messages)
    assert usage.total_input_tokens == 130
    assert usage.total_output_tokens == 90
    assert usage.total_cache_read_tokens == 20
    assert usage.total_cache_creation_tokens == 10
    assert usage.total_requests == 2
    assert usage.total_duration_ms == 1500
    assert usage.total_cost_usd == pytest.approx(0.0163, rel=0, abs=1e-9)


def test_rebuild_session_usage_skips_non_usage_assistant_messages() -> None:
    reset_session_usage()
    messages = [
        create_assistant_message("status only"),
        create_assistant_message("has model", model="gpt-5"),
    ]

    usage = rebuild_session_usage(messages)
    assert usage.total_requests == 1
    assert "gpt-5" in usage.models
    assert usage.models["gpt-5"].requests == 1

    # Verify the global snapshot matches rebuilt result.
    global_snapshot = get_session_usage()
    assert global_snapshot.total_requests == usage.total_requests
