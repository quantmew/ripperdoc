# Ripperdoc Python SDK (headless)

Call Ripperdoc directly from Python without launching the terminal UI. A one-shot `query` helper for ad-hoc calls and a session-based `RipperdocClient` for conversations that keep history.

## Quick start (one-off)

```python
import asyncio
from ripperdoc.sdk import query, RipperdocOptions
from ripperdoc.utils.messages import AssistantMessage, ProgressMessage


async def main():
    options = RipperdocOptions(
        yolo_mode=True,  # set True to skip interactive permission prompts
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
    async with RipperdocClient(RipperdocOptions(yolo_mode=True)) as client:
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

- `yolo_mode` (defaults to False, meaning permission prompts are enabled); set to True to skip prompts or provide a custom `permission_checker`.
- `allowed_tools` / `disallowed_tools` limit which tools the agent can call.
- `cwd` pins the working directory for the session.
- `additional_instructions` appends text to the system prompt; use `system_prompt` to fully override it.
- `model`, `max_thinking_tokens`, `context`, `permission_checker` pass directly into the core query pipeline.

Messages emitted by the SDK are the existing `UserMessage`, `AssistantMessage`, and `ProgressMessage` classes from `ripperdoc.utils.messages`. API keys and model profiles are loaded the same way as the CLI (environment variables or `~/.ripperdoc.json`).

## Thinking / reasoning mode

- Set `max_thinking_tokens` on `RipperdocOptions` (or QueryContext) to request a thinking-capable run.
- OpenAI-compatible models: DeepSeek adds `thinking={"type": "enabled"}`, GPT-5/OpenRouter reasoning models send a `reasoning` effort hint, and qwen/dashscope toggles `enable_thinking` when thinking is requested.
- Gemini: maps `max_thinking_tokens` to `thinking_config` (`thinking_budget` for 2.5, `thinking_level` for 3) and turns on `include_thoughts` so summaries come back.
- Anthropic: when `max_thinking_tokens > 0`, sends `thinking={type: enabled, budget_tokens: N}` and preserves returned `thinking`/`redacted_thinking` blocks; the UI renders a dim “Thinking” preview.
- Reasoning traces (`reasoning_content` / `reasoning_details`) are stored on assistant messages and replayed automatically in the next turn so tool-calling loops keep the chain-of-thought intact.
- You can force a thinking protocol by setting `thinking_mode` on a model profile (`deepseek`, `openrouter`, `qwen`, `gemini_openai`, `openai`) to bypass heuristics.
