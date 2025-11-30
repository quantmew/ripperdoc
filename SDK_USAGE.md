# Ripperdoc Python SDK (headless)

Call Ripperdoc directly from Python without launching the terminal UI. A one-shot `query` helper for ad-hoc calls and a session-based `RipperdocClient` for conversations that keep history.

## Quick start (one-off)

```python
import asyncio
from ripperdoc.sdk import query, RipperdocOptions
from ripperdoc.utils.messages import AssistantMessage, ProgressMessage


async def main():
    options = RipperdocOptions(
        safe_mode=False,  # default is False to avoid interactive permission prompts
        allowed_tools=["Bash", "View", "Glob"],
        cwd="/path/to/project",
    )

    async for msg in query("List Python files", options=options):
        if isinstance(msg, AssistantMessage):
            print(msg.message.content)
        elif isinstance(msg, ProgressMessage):
            print(f"[progress] {msg.content}")


asyncio.run(main())
```

## Long-running sessions

```python
import asyncio
from ripperdoc.sdk import RipperdocClient, RipperdocOptions
from ripperdoc.utils.messages import AssistantMessage


async def main():
    async with RipperdocClient(RipperdocOptions(safe_mode=False)) as client:
        await client.query("Summarize README.md")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                print(msg.message.content)

        await client.query("Where is the CLI entrypoint defined?")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                print(msg.message.content)


asyncio.run(main())
```

## Options you can tweak

- `safe_mode` (defaults to False here to avoid blocking on prompts); set to True to reuse the CLI-style permission prompts or provide a custom `permission_checker`.
- `allowed_tools` / `disallowed_tools` limit which tools the agent can call.
- `cwd` pins the working directory for the session.
- `additional_instructions` appends text to the system prompt; use `system_prompt` to fully override it.
- `model`, `max_thinking_tokens`, `context`, `permission_checker` pass directly into the core query pipeline.

Messages emitted by the SDK are the existing `UserMessage`, `AssistantMessage`, and `ProgressMessage` classes from `ripperdoc.utils.messages`. API keys and model profiles are loaded the same way as the CLI (environment variables or `~/.ripperdoc.json`).
