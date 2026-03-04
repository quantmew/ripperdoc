from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ripperdoc.cli.ui.rich_ui.input import _handle_tab_completion


@dataclass
class _FakeCompleteState:
    completions: list[Any]
    current_completion: Any | None = None


class _FakeBuffer:
    def __init__(self, *, complete_state: _FakeCompleteState | None, start_states: list[_FakeCompleteState | None] | None = None):
        self.complete_state = complete_state
        self._start_states = list(start_states or [])
        self.applied: list[Any] = []
        self.start_calls = 0

    def start_completion(self, *, select_first: bool) -> None:
        assert select_first is True
        self.start_calls += 1
        if self._start_states:
            self.complete_state = self._start_states.pop(0)

    def apply_completion(self, completion: Any) -> None:
        self.applied.append(completion)


def test_tab_completion_applies_first_when_multiple_and_no_current() -> None:
    first = object()
    second = object()
    buf = _FakeBuffer(complete_state=_FakeCompleteState(completions=[first, second], current_completion=None))

    _handle_tab_completion(buf)

    assert buf.applied == [first]
    assert buf.start_calls == 0


def test_tab_completion_applies_first_after_start_completion() -> None:
    first = object()
    second = object()
    buf = _FakeBuffer(
        complete_state=None,
        start_states=[_FakeCompleteState(completions=[first, second], current_completion=None)],
    )

    _handle_tab_completion(buf)

    assert buf.applied == [first]
    assert buf.start_calls == 1


def test_tab_completion_retries_when_no_entries_available() -> None:
    buf = _FakeBuffer(
        complete_state=None,
        start_states=[
            _FakeCompleteState(completions=[], current_completion=None),
            _FakeCompleteState(completions=[], current_completion=None),
        ],
    )

    _handle_tab_completion(buf)

    assert buf.applied == []
    assert buf.start_calls == 2
