"""Ensure background bash completions enqueue conversation notices."""

import asyncio

import pytest

from ripperdoc.core.tool import ToolUseContext
from ripperdoc.tools.background_shell import get_background_status
from ripperdoc.tools.bash_tool import BashTool, BashToolInput
from ripperdoc.utils.messaging.pending_messages import PendingMessageQueue


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_background_completion_enqueues_notification():
    queue = PendingMessageQueue()
    context = ToolUseContext(
        pending_message_queue=queue,
        task_notification_queue=queue,
    )
    tool = BashTool()

    input_data = BashToolInput(
        command="sleep 0.05 && echo done", run_in_background=True, timeout=2000
    )

    outputs = []
    async for result in tool.call(input_data, context):
        outputs.append(result)

    task_id = None
    for result in outputs:
        data = getattr(result, "data", None)
        task_id = getattr(data, "background_task_id", None)
        if task_id:
            break

    assert task_id is not None, "Background task id should be returned"

    # Wait for the task to finish so the completion callback can enqueue a notice.
    status: dict = {}
    for _ in range(50):
        status = get_background_status(task_id, consume=False)
        if status.get("status") != "running":
            break
        await asyncio.sleep(0.05)

    assert status.get("status") != "running"

    pending = []
    for _ in range(10):
        if queue.has_messages():
            pending = queue.drain()
            break
        await asyncio.sleep(0.05)

    assert pending, "Expected queued background completion notice"
    notice_content = getattr(getattr(pending[0], "message", None), "content", "")
    assert isinstance(notice_content, str)
    assert task_id in notice_content
    assert "<task-notification>" in notice_content
    metadata = getattr(getattr(pending[0], "message", None), "metadata", {})
    assert metadata.get("notification_type") == "task_notification"
    assert metadata.get("task_id") == task_id


@pytest.mark.asyncio
@pytest.mark.usefixtures("reset_background_shell")
async def test_background_completion_uses_task_notification_queue_when_provided():
    local_queue = PendingMessageQueue()
    parent_queue = PendingMessageQueue()
    context = ToolUseContext(
        pending_message_queue=local_queue,
        task_notification_queue=parent_queue,
    )
    tool = BashTool()

    input_data = BashToolInput(command="sleep 0.05 && echo done", run_in_background=True, timeout=2000)
    outputs = []
    async for result in tool.call(input_data, context):
        outputs.append(result)

    task_id = None
    for result in outputs:
        data = getattr(result, "data", None)
        task_id = getattr(data, "background_task_id", None)
        if task_id:
            break
    assert task_id is not None

    status: dict = {}
    for _ in range(50):
        status = get_background_status(task_id, consume=False)
        if status.get("status") != "running":
            break
        await asyncio.sleep(0.05)
    assert status.get("status") != "running"

    pending_parent = []
    for _ in range(10):
        if parent_queue.has_messages():
            pending_parent = parent_queue.drain()
            break
        await asyncio.sleep(0.05)
    assert pending_parent, "Expected completion notice on parent task notification queue"
    assert not local_queue.has_messages(), "Local queue should not receive task notification directly"
