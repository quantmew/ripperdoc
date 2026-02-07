"""Tests for team persistence and team/task integration."""

from __future__ import annotations

import asyncio

from ripperdoc.core.tool import ToolUseContext
from ripperdoc.tools.task_graph_tool import TaskCreateInput, TaskCreateTool, TaskUpdateInput, TaskUpdateTool
from ripperdoc.tools.team_tool import (
    SendMessageInput,
    SendMessageTool,
    TeamCreateInput,
    TeamCreateTool,
    TeamDeleteInput,
    TeamDeleteTool,
)
from ripperdoc.utils.teams import TeamMember, clear_active_team_name, list_team_messages, upsert_team_member


def test_team_tools_create_and_send_message(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_tool = TeamCreateTool()
    send_tool = SendMessageTool()
    context = ToolUseContext(agent_id="team-lead")

    async def _run() -> None:
        created = None
        async for output in create_tool.call(
            TeamCreateInput(
                team_name="alpha",
                description="auth refactor",
                agent_type="general-purpose",
            ),
            context,
        ):
            created = output

        assert created is not None
        assert created.data.team_name == "alpha"
        assert created.data.lead_agent_id == "team-lead@alpha"

        sent = None
        async for output in send_tool.call(
            SendMessageInput(
                type="message",
                recipient="dev-a",
                content="Please pick task 1",
                summary="Please pick up task one now",
            ),
            context,
        ):
            sent = output

        assert sent is not None
        assert sent.data.success is True
        assert sent.data.routing is not None
        assert sent.data.routing.target == "@dev-a"

    asyncio.run(_run())

    messages = list_team_messages("alpha", limit=10)
    assert messages
    assert messages[-1].message_type == "message"


def test_task_update_owner_emits_assignment_message(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    team_create = TeamCreateTool()
    task_create = TaskCreateTool()
    task_update = TaskUpdateTool()
    context = ToolUseContext(agent_id="team-lead")

    async def _run() -> None:
        async for _ in team_create.call(
            TeamCreateInput(team_name="alpha", description="demo"),
            context,
        ):
            pass

        created = None
        async for output in task_create.call(
            TaskCreateInput(subject="do work", description="implement feature"),
            context,
        ):
            created = output

        assert created is not None
        assert created.data.task is not None
        task_id = created.data.task.id

        async for _ in task_update.call(
            TaskUpdateInput(task_id=task_id, owner="dev-a"),
            context,
        ):
            pass

    asyncio.run(_run())

    messages = list_team_messages("alpha", limit=20)
    assignment_messages = [msg for msg in messages if msg.message_type == "task_assignment"]
    assert assignment_messages
    assert assignment_messages[-1].recipients == ["dev-a"]


def test_send_message_summary_word_count_validation():
    send_tool = SendMessageTool()

    async def _run() -> None:
        valid_message = await send_tool.validate_input(
            SendMessageInput(
                type="message",
                recipient="dev-a",
                content="Please investigate auth failures.",
                summary="Please investigate auth regression failures first",
            )
        )
        assert valid_message.result is True

        invalid_short = await send_tool.validate_input(
            SendMessageInput(
                type="message",
                recipient="dev-a",
                content="Please investigate auth failures.",
                summary="auth regression fix now",
            )
        )
        assert invalid_short.result is False
        assert "5-10 words" in (invalid_short.message or "")

        invalid_long = await send_tool.validate_input(
            SendMessageInput(
                type="broadcast",
                content="Pause all commits while we fix build.",
                summary="This summary intentionally contains too many words for strict validation checks today",
            )
        )
        assert invalid_long.result is False
        assert "5-10 words" in (invalid_long.message or "")

    asyncio.run(_run())


def test_team_create_allows_new_team_when_only_singleton_team_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RIPPERDOC_TEAM_NAME", raising=False)

    create_tool = TeamCreateTool()

    async def _run() -> None:
        context_a = ToolUseContext(agent_id="lead-a")
        async for _ in create_tool.call(TeamCreateInput(team_name="alpha"), context_a):
            pass

        clear_active_team_name("alpha")

        created = None
        context_b = ToolUseContext(agent_id="lead-b")
        async for output in create_tool.call(TeamCreateInput(team_name="beta"), context_b):
            created = output

        assert created is not None
        assert created.data.team_name == "beta"

    asyncio.run(_run())


def test_team_delete_requires_no_active_members(tmp_path, monkeypatch):
    monkeypatch.setattr("ripperdoc.utils.tasks.Path.home", lambda: tmp_path)
    monkeypatch.setattr("ripperdoc.utils.teams.Path.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    create_tool = TeamCreateTool()
    delete_tool = TeamDeleteTool()
    context = ToolUseContext(agent_id="team-lead")

    async def _run() -> None:
        async for _ in create_tool.call(TeamCreateInput(team_name="alpha"), context):
            pass

        upsert_team_member(
            "alpha",
            TeamMember(name="dev-a", agent_type="general-purpose", active=True),
        )

        failed = None
        async for output in delete_tool.call(TeamDeleteInput(), context):
            failed = output

        assert failed is not None
        assert failed.data.success is False
        assert "active member" in failed.data.message

        upsert_team_member(
            "alpha",
            TeamMember(name="dev-a", agent_type="general-purpose", active=False),
        )

        deleted = None
        async for output in delete_tool.call(TeamDeleteInput(), context):
            deleted = output

        assert deleted is not None
        assert deleted.data.success is True

    asyncio.run(_run())


def test_team_tool_descriptions_and_examples():
    async def _run() -> None:
        create_tool = TeamCreateTool()
        send_tool = SendMessageTool()

        create_desc = await create_tool.description()
        send_desc = await send_tool.description()

        assert "team" in create_desc.lower()
        assert "message" in send_desc.lower() or "protocol" in send_desc.lower()
        assert len(create_tool.input_examples()) > 0
        assert len(send_tool.input_examples()) > 0

    asyncio.run(_run())
