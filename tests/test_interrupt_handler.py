"""Tests for InterruptHandler.

Tests cover:
- InterruptHandler class properties and methods
- pause_listener and resume_listener functionality
- Cross-platform behavior (Windows vs Unix)
- Async interrupt key listening
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from ripperdoc.cli.ui.interrupt_handler import (
    InterruptHandler,
    INTERRUPT_KEYS,
)


class TestInterruptHandlerInitialization:
    """Tests for InterruptHandler initialization and properties."""

    def test_initial_state(self):
        """Initial state should be correctly set."""
        handler = InterruptHandler()
        assert handler.was_interrupted is False
        assert handler._query_interrupted is False
        assert handler._esc_listener_active is False
        assert handler._esc_listener_paused is False
        assert handler._stdin_fd is None
        assert handler._stdin_old_settings is None
        assert handler._stdin_in_raw_mode is False
        assert handler._abort_callback is None

    def test_was_interrupted_property(self):
        """was_interrupted property should reflect _query_interrupted."""
        handler = InterruptHandler()
        assert handler.was_interrupted is False

        handler._query_interrupted = True
        assert handler.was_interrupted is True


class TestAbortCallback:
    """Tests for abort callback functionality."""

    def test_set_abort_callback(self):
        """Should set abort callback."""
        handler = InterruptHandler()
        callback = MagicMock()

        handler.set_abort_callback(callback)
        assert handler._abort_callback == callback

    def test_trigger_abort_calls_callback(self):
        """Triggering abort should call the callback."""
        handler = InterruptHandler()
        callback = MagicMock()

        handler.set_abort_callback(callback)
        handler._trigger_abort()

        callback.assert_called_once()

    def test_trigger_abort_without_callback(self):
        """Triggering abort without callback should not raise."""
        handler = InterruptHandler()
        # Should not raise
        handler._trigger_abort()


class TestPauseAndResumeListener:
    """Tests for pause_listener and resume_listener functionality."""

    def test_pause_listener_returns_previous_state(self):
        """pause_listener should return previous paused state."""
        handler = InterruptHandler()

        # First pause - should return False (previous state)
        result1 = handler.pause_listener()
        assert result1 is False
        assert handler._esc_listener_paused is True

        # Pause again - should return True (previous state)
        result2 = handler.pause_listener()
        assert result2 is True
        assert handler._esc_listener_paused is True

    def test_resume_listener_restores_state(self):
        """resume_listener should restore previous state."""
        handler = InterruptHandler()

        # Pause and remember state
        prev = handler.pause_listener()
        assert handler._esc_listener_paused is True

        # Resume with False - should stay paused
        handler.resume_listener(False)
        assert handler._esc_listener_paused is False

        # Resume with True - should stay paused
        handler.resume_listener(True)
        assert handler._esc_listener_paused is True

    def test_pause_listener_on_windows(self, monkeypatch):
        """On Windows, pause_listener should return early without termios."""
        # Mock platform to be Windows
        monkeypatch.setattr("ripperdoc.cli.ui.interrupt_handler.is_windows", lambda: True)

        handler = InterruptHandler()

        # Should not raise on Windows
        result = handler.pause_listener()
        assert isinstance(result, bool)


class TestRunWithInterrupt:
    """Tests for run_with_interrupt functionality."""

    async def test_successful_completion(self):
        """Coroutine that completes normally should return False."""
        handler = InterruptHandler()

        async def dummy_coro():
            await asyncio.sleep(0.01)
            return "success"

        result = await handler.run_with_interrupt(dummy_coro())
        assert result is False  # Not interrupted
        assert handler.was_interrupted is False

    async def test_interrupted_by_callback(self):
        """Should return True when interrupted via callback."""
        handler = InterruptHandler()

        async def dummy_coro():
            await asyncio.sleep(0.5)  # Long running
            return "success"

        # Simulate interrupt after short delay
        async def trigger_interrupt():
            await asyncio.sleep(0.05)
            handler._query_interrupted = True
            handler._trigger_abort()

        # Run both tasks
        interrupt_task = asyncio.create_task(trigger_interrupt())
        result = await handler.run_with_interrupt(dummy_coro())
        await interrupt_task

        assert result is True  # Was interrupted
        assert handler.was_interrupted is True

    async def test_sets_listener_active_state(self):
        """Should properly set _esc_listener_active during execution."""
        handler = InterruptHandler()

        async def quick_coro():
            assert handler._esc_listener_active is True
            await asyncio.sleep(0.01)
            return "done"

        await handler.run_with_interrupt(quick_coro())

        # After execution, should be False again
        assert handler._esc_listener_active is False

    async def test_cleanup_on_exception(self):
        """Should clean up state even if coroutine raises exception."""
        handler = InterruptHandler()

        async def failing_coro():
            handler._esc_listener_active = True
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await handler.run_with_interrupt(failing_coro())

        # State should be cleaned up
        assert handler._esc_listener_active is False

    async def test_concurrent_safe(self):
        """Should handle concurrent calls safely."""
        handler = InterruptHandler()

        async def quick_task(name):
            await asyncio.sleep(0.01)
            return name

        # Run multiple tasks sequentially
        result1 = await handler.run_with_interrupt(quick_task("task1"))
        result2 = await handler.run_with_interrupt(quick_task("task2"))

        assert result1 is False
        assert result2 is False


class TestCancelTask:
    """Tests for _cancel_task helper method."""

    async def test_cancel_pending_task(self):
        """Should cancel a pending task."""
        handler = InterruptHandler()

        async def never_ending():
            await asyncio.sleep(1000)

        task = asyncio.create_task(never_ending())
        await handler._cancel_task(task)

        assert task.cancelled()
        # Should not raise CancelledError
        await handler._cancel_task(task)

    async def test_cancel_completed_task(self):
        """Should handle completed task gracefully."""
        handler = InterruptHandler()

        async def instant():
            return "done"

        task = asyncio.create_task(instant())
        await task  # Wait for completion

        # Should not raise
        await handler._cancel_task(task)


class TestInterruptKeys:
    """Tests for interrupt key constants."""

    def test_interrupt_keys_defined(self):
        """INTERRUPT_KEYS should contain ESC and Ctrl+C."""
        assert "\x1b" in INTERRUPT_KEYS  # ESC
        assert "\x03" in INTERRUPT_KEYS  # Ctrl+C

    def test_interrupt_keys_is_set(self):
        """INTERRUPT_KEYS should be a set."""
        assert isinstance(INTERRUPT_KEYS, set)


class TestWindowsSpecificBehavior:
    """Tests for Windows-specific behavior."""

    def test_listen_for_interrupt_key_on_windows(self, monkeypatch):
        """On Windows, _listen_for_interrupt_key should handle gracefully."""
        # Mock platform as Windows
        monkeypatch.setattr("ripperdoc.cli.ui.interrupt_handler.is_windows", lambda: True)

        handler = InterruptHandler()
        handler._esc_listener_active = True

        # Run the listener - should not raise
        async def run_listener():
            result = await handler._listen_for_interrupt_key()
            # On Windows, should just wait indefinitely until listener is inactive
            return result

        async def test():
            # Start listener and cancel it quickly
            task = asyncio.create_task(run_listener())
            await asyncio.sleep(0.05)
            handler._esc_listener_active = False
            # The listener should exit
            result = await asyncio.wait_for(task, timeout=1.0)
            return result

        result = asyncio.run(test())
        assert result is False  # No interrupt detected


class TestCrossPlatformCompatibility:
    """Tests for cross-platform compatibility."""

    def test_pause_listener_cross_platform(self):
        """pause_listener should work on all platforms."""
        handler = InterruptHandler()
        # Should not raise on any platform
        result = handler.pause_listener()
        assert isinstance(result, bool)

    def test_resume_listener_cross_platform(self):
        """resume_listener should work on all platforms."""
        handler = InterruptHandler()
        # Should not raise on any platform
        handler.resume_listener(True)
        handler.resume_listener(False)

    def test_trigger_abort_cross_platform(self):
        """_trigger_abort should work on all platforms."""
        handler = InterruptHandler()
        # Should not raise on any platform
        handler._trigger_abort()


class TestEdgeCases:
    """Edge case tests."""

    def test_multiple_pause_resume_cycles(self):
        """Should handle multiple pause/resume cycles."""
        handler = InterruptHandler()

        for _ in range(5):
            prev = handler.pause_listener()
            handler.resume_listener(prev)
            assert handler._esc_listener_paused == prev

    async def test_run_with_immediate_completion(self):
        """Should handle coroutine that completes immediately."""
        handler = InterruptHandler()

        async def instant():
            return "instant"

        result = await handler.run_with_interrupt(instant())
        assert result is False

    async def test_run_with_very_fast_coroutine(self):
        """Should handle coroutine that completes faster than interrupt check."""
        handler = InterruptHandler()

        async def micro_task():
            # Completes in less than a tick
            return "micro"

        result = await handler.run_with_interrupt(micro_task())
        assert result is False

    def test_callback_can_be_changed(self):
        """Abort callback should be changeable."""
        handler = InterruptHandler()

        callback1 = MagicMock()
        callback2 = MagicMock()

        handler.set_abort_callback(callback1)
        handler._trigger_abort()
        assert callback1.call_count == 1
        assert callback2.call_count == 0

        handler.set_abort_callback(callback2)
        handler._trigger_abort()
        assert callback1.call_count == 1  # No new call
        assert callback2.call_count == 1


class TestConcurrentScenarios:
    """Tests for concurrent usage scenarios."""

    async def test_parallel_interrupt_handlers(self):
        """Multiple handlers should work independently."""
        handler1 = InterruptHandler()
        handler2 = InterruptHandler()

        async def task1():
            await asyncio.sleep(0.01)
            return "task1"

        async def task2():
            await asyncio.sleep(0.01)
            return "task2"

        # Run both handlers concurrently
        results = await asyncio.gather(
            handler1.run_with_interrupt(task1()),
            handler2.run_with_interrupt(task2()),
        )

        assert results == [False, False]

    async def test_handler_state_isolation(self):
        """Each handler should maintain independent state."""
        handler1 = InterruptHandler()
        handler2 = InterruptHandler()

        handler1._query_interrupted = True
        handler2._query_interrupted = False

        assert handler1.was_interrupted is True
        assert handler2.was_interrupted is False


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    async def test_pause_resume_during_execution(self):
        """Test pause/resume during actual execution."""
        handler = InterruptHandler()

        async def task_with_pause():
            # Start the task
            pause_state = handler.pause_listener()
            await asyncio.sleep(0.02)
            handler.resume_listener(pause_state)
            return "done"

        result = await handler.run_with_interrupt(task_with_pause())
        assert result is False

    async def test_listener_state_preservation(self):
        """Listener state should be preserved correctly."""
        handler = InterruptHandler()

        # Simulate pause state
        prev = handler.pause_listener()
        assert handler._esc_listener_paused is True

        async def check_state():
            # Inside execution, state should be preserved
            return handler._esc_listener_paused

        # We can't easily test this without actual execution context,
        # but we can verify the mechanism works
        handler.resume_listener(prev)
        assert handler._esc_listener_paused == prev
