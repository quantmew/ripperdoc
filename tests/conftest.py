"""Pytest configuration and fixtures for all tests."""

import pytest

from ripperdoc.tools import background_shell


@pytest.fixture(scope="session", autouse=True)
def cleanup_background_shell_after_all_tests():
    """Ensure the background shell is cleaned up after all tests complete.

    This fixture runs automatically for all test sessions and ensures that
    any background shell threads created during testing are properly shut down.
    Without this, pytest may hang after tests complete because the background
    loop thread (which is non-daemon) keeps the process alive.
    """
    # Run all tests
    yield

    # Cleanup after all tests complete
    background_shell.shutdown_background_shell()


@pytest.fixture(autouse=False)
def reset_background_shell():
    """Reset background shell state before and after a test.

    Use this fixture explicitly for tests that need a clean background shell state.
    Example: @pytest.mark.usefixtures("reset_background_shell")
    """
    # Ensure clean state before test
    background_shell.reset_background_shell_for_testing()

    yield

    # Ensure clean state after test
    background_shell.reset_background_shell_for_testing()
