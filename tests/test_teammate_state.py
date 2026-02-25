"""Tests for teammate state management and idle notification functionality."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from ripperdoc.utils.teams import (
    TeamMember,
    create_team,
    list_team_messages,
    upsert_team_member,
)
from ripperdoc.utils.teammate_state import (
    IdleReason,
    TeammateStatus,
    InProcessTeammateState,
    _TEAMMATE_STATES,
    _TEAMMATE_STATES_LOCK,
    create_teammate_state,
    get_teammate_state,
    get_teammate_state_by_agent_id,
    list_running_teammates,
    set_teammate_idle,
    set_teammate_active,
    inject_user_message,
    pop_pending_user_message,
    request_teammate_shutdown,
    add_idle_callback,
    complete_teammate,
    cleanup_teammate_state,
    has_running_teammates,
    has_active_teammates,
    format_teammate_message,
)


@pytest.fixture
def temp_home(monkeypatch, tmp_path):
    """Set up a temporary home directory for team files."""
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    # Clear global teammate state before each test
    with _TEAMMATE_STATES_LOCK:
        _TEAMMATE_STATES.clear()

    yield tmp_path

    # Clear global teammate state after each test
    with _TEAMMATE_STATES_LOCK:
        _TEAMMATE_STATES.clear()


class TestTeammateStateCreation:
    """Tests for creating and managing teammate state."""

    def test_create_teammate_state_basic(self, temp_home):
        """Test basic teammate state creation."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="researcher",
            team_name="test-team",
            prompt="Research the codebase",
        )

        assert state.id.startswith("teammate_")
        assert state.identity.agent_name == "researcher"
        assert state.identity.team_name == "test-team"
        assert state.status == TeammateStatus.RUNNING
        assert state.is_idle is False
        assert state.permission_mode == "default"

    def test_create_teammate_state_with_plan_mode(self, temp_home):
        """Test teammate state creation with plan mode required."""
        create_team(name="plan-team")

        state = create_teammate_state(
            agent_name="planner",
            team_name="plan-team",
            prompt="Create a plan",
            plan_mode_required=True,
        )

        assert state.permission_mode == "plan"

    def test_get_teammate_state(self, temp_home):
        """Test retrieving teammate state by ID."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        retrieved = get_teammate_state(state.id)
        assert retrieved is not None
        assert retrieved.id == state.id

    def test_get_teammate_state_by_agent_id(self, temp_home):
        """Test finding teammate state by agent ID."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="unique-worker",
            team_name="test-team",
            prompt="Do work",
        )

        found = get_teammate_state_by_agent_id("unique-worker@test-team")
        assert found is not None
        assert found.id == state.id

    def test_list_running_teammates(self, temp_home):
        """Test listing running teammates."""
        create_team(name="team-a")
        create_team(name="team-b")

        state1 = create_teammate_state(
            agent_name="worker1",
            team_name="team-a",
            prompt="Work 1",
        )
        state2 = create_teammate_state(
            agent_name="worker2",
            team_name="team-a",
            prompt="Work 2",
        )
        state3 = create_teammate_state(
            agent_name="worker3",
            team_name="team-b",
            prompt="Work 3",
        )

        all_running = list_running_teammates()
        assert len(all_running) == 3

        team_a_running = list_running_teammates(team_name="team-a")
        assert len(team_a_running) == 2

        team_b_running = list_running_teammates(team_name="team-b")
        assert len(team_b_running) == 1

        # Clean up
        cleanup_teammate_state(state1.id)
        cleanup_teammate_state(state2.id)
        cleanup_teammate_state(state3.id)


class TestIdleStateManagement:
    """Tests for idle state management."""

    def test_set_teammate_idle(self, temp_home):
        """Test setting teammate to idle state."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        assert state.is_idle is False

        result = set_teammate_idle(
            state.id,
            idle_reason=IdleReason.AVAILABLE,
            summary="Work completed",
        )

        assert result is True
        assert state.is_idle is True
        assert state.idle_since is not None

        # Check idle notification was sent
        messages = list_team_messages("test-team", limit=5)
        idle_msgs = [m for m in messages if m.message_type == "idle_notification"]
        assert len(idle_msgs) == 1

        payload = json.loads(idle_msgs[0].content)
        assert payload["idleReason"] == "available"
        assert payload["summary"] == "Work completed"

        cleanup_teammate_state(state.id)

    def test_set_teammate_active(self, temp_home):
        """Test setting teammate back to active state."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        set_teammate_idle(state.id, idle_reason=IdleReason.AVAILABLE)
        assert state.is_idle is True

        result = set_teammate_active(state.id)
        assert result is True
        assert state.is_idle is False

        cleanup_teammate_state(state.id)

    def test_idle_callbacks_executed(self, temp_home):
        """Test that idle callbacks are executed when going idle."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        callback_called = []

        def on_idle():
            callback_called.append(True)

        add_idle_callback(state.id, on_idle)

        set_teammate_idle(state.id, idle_reason=IdleReason.AVAILABLE)

        assert len(callback_called) == 1
        assert len(state.on_idle_callbacks) == 0  # Callbacks cleared after execution

        cleanup_teammate_state(state.id)

    def test_has_active_teammates(self, temp_home):
        """Test checking for active (non-idle) teammates."""
        create_team(name="test-team")

        state1 = create_teammate_state(
            agent_name="worker1",
            team_name="test-team",
            prompt="Work 1",
        )
        state2 = create_teammate_state(
            agent_name="worker2",
            team_name="test-team",
            prompt="Work 2",
        )

        assert has_active_teammates("test-team") is True

        set_teammate_idle(state1.id, idle_reason=IdleReason.AVAILABLE)
        assert has_active_teammates("test-team") is True  # state2 still active

        set_teammate_idle(state2.id, idle_reason=IdleReason.AVAILABLE)
        assert has_active_teammates("test-team") is False

        cleanup_teammate_state(state1.id)
        cleanup_teammate_state(state2.id)


class TestMessageInjection:
    """Tests for pending user message queue."""

    def test_inject_user_message(self, temp_home):
        """Test injecting a message into teammate's queue."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        set_teammate_idle(state.id, idle_reason=IdleReason.AVAILABLE)
        assert state.is_idle is True

        result = inject_user_message(state.id, "New task for you!")
        assert result is True
        assert len(state.pending_user_messages) == 1
        assert state.is_idle is False  # Woken up by message

        cleanup_teammate_state(state.id)

    def test_pop_pending_user_message(self, temp_home):
        """Test popping messages from the queue."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        inject_user_message(state.id, "Message 1")
        inject_user_message(state.id, "Message 2")

        msg1 = pop_pending_user_message(state.id)
        assert msg1 == "Message 1"

        msg2 = pop_pending_user_message(state.id)
        assert msg2 == "Message 2"

        msg3 = pop_pending_user_message(state.id)
        assert msg3 is None

        cleanup_teammate_state(state.id)


class TestShutdownProtocol:
    """Tests for shutdown request handling."""

    def test_request_teammate_shutdown(self, temp_home):
        """Test requesting a teammate to shut down."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        assert state.shutdown_requested is False

        result = request_teammate_shutdown(state.id, reason="Work complete")
        assert result is True
        assert state.shutdown_requested is True
        assert state.shutdown_request_id is not None

        # Second request should still return True but not change state
        result2 = request_teammate_shutdown(state.id)
        assert result2 is True

        cleanup_teammate_state(state.id)

    def test_complete_teammate(self, temp_home):
        """Test marking a teammate as completed."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        result = complete_teammate(
            state.id,
            status=TeammateStatus.COMPLETED,
            result_text="All done!",
        )

        assert result is True
        assert state.status == TeammateStatus.COMPLETED
        assert state.end_time is not None
        assert state.result_text == "All done!"
        assert state.is_idle is True

        cleanup_teammate_state(state.id)


class TestIdleReasons:
    """Tests for different idle reasons."""

    @pytest.mark.parametrize("reason,expected", [
        (IdleReason.AVAILABLE, "available"),
        (IdleReason.INTERRUPTED, "interrupted"),
        (IdleReason.FAILED, "failed"),
        (IdleReason.SHUTDOWN, "shutdown"),
    ])
    def test_idle_reasons_in_notification(self, temp_home, reason, expected):
        """Test that idle reasons are correctly included in notifications."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
        )

        set_teammate_idle(state.id, idle_reason=reason, summary="Test summary")

        messages = list_team_messages("test-team", limit=5)
        idle_msgs = [m for m in messages if m.message_type == "idle_notification"]
        assert len(idle_msgs) == 1

        payload = json.loads(idle_msgs[0].content)
        assert payload["idleReason"] == expected

        cleanup_teammate_state(state.id)


class TestFormatTeammateMessage:
    """Tests for message formatting."""

    def test_format_basic_message(self):
        """Test formatting a basic teammate message."""
        msg = format_teammate_message(
            from_name="team-lead",
            message="Please work on task 1",
        )

        assert '<teammate_message from="team-lead">' in msg
        assert "Please work on task 1" in msg
        assert "</teammate_message>" in msg

    def test_format_message_with_color(self):
        """Test formatting a message with color."""
        msg = format_teammate_message(
            from_name="team-lead",
            message="Test message",
            color="#ff0000",
        )

        assert 'color="#ff0000"' in msg

    def test_format_message_with_summary(self):
        """Test formatting a message with summary."""
        msg = format_teammate_message(
            from_name="team-lead",
            message="Full message content here",
            summary="Brief summary",
        )

        assert 'summary="Brief summary"' in msg

    def test_format_message_escapes_quotes_in_summary(self):
        """Test that quotes in summary are escaped."""
        msg = format_teammate_message(
            from_name="team-lead",
            message="Content",
            summary='He said "hello"',
        )

        assert 'summary="He said \\"hello\\""' in msg


class TestTeammateStateSnapshot:
    """Tests for state snapshot functionality."""

    def test_to_snapshot(self, temp_home):
        """Test generating a state snapshot."""
        create_team(name="test-team")

        state = create_teammate_state(
            agent_name="worker",
            team_name="test-team",
            prompt="Do work",
            model="main",
        )

        inject_user_message(state.id, "Test message")
        state.last_reported_tool_count = 5

        snapshot = state.to_snapshot()

        assert snapshot["id"] == state.id
        assert snapshot["identity"]["agent_name"] == "worker"
        assert snapshot["identity"]["team_name"] == "test-team"
        assert snapshot["status"] == "running"
        assert snapshot["is_idle"] is False
        assert snapshot["pending_messages_count"] == 1
        assert snapshot["tool_count"] == 5
        assert snapshot["permission_mode"] == "default"

        cleanup_teammate_state(state.id)
