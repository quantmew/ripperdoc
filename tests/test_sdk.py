"""Tests for the headless SDK."""

import asyncio

from ripperdoc.sdk import RipperdocClient, RipperdocOptions, query as sdk_query
from ripperdoc.utils.messages import create_assistant_message


async def _fake_runner(messages, system_prompt, context, query_context, permission_checker):
    del messages, system_prompt, context, query_context, permission_checker
    yield create_assistant_message("OK")


def test_tool_filtering_respects_allowed_names():
    async def _run():
        options = RipperdocOptions(
            allowed_tools=["Bash", "Read", "Task"],
            yolo_mode=True,
        )
        client = RipperdocClient(options=options, query_runner=_fake_runner)
        await client.connect()

        names = {tool.name for tool in client.tools}
        assert names == {"Bash", "Read", "Task"}

        task_tool = next(tool for tool in client.tools if tool.name == "Task")
        base_names = {tool.name for tool in task_tool._available_tools_provider()}  # type: ignore[attr-defined]
        assert base_names == {"Bash", "Read"}

        await client.disconnect()

    asyncio.run(_run())


def test_query_helper_streams_messages():
    async def _run():
        options = RipperdocOptions(yolo_mode=True)
        messages = []

        async for message in sdk_query("hello", options=options, query_runner=_fake_runner):
            messages.append(message)

        assert messages
        assert messages[0].message.content == "OK"

    asyncio.run(_run())
