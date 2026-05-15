"""Sleep tool prompt."""

SLEEP_PROMPT = """Wait for a specified duration. The user can interrupt the sleep at any time.

Use this when the user tells you to sleep or rest, when you have nothing to do, or when you're waiting for something.

You can call this concurrently with other tools - it won't interfere with them.

Prefer this over `Bash(sleep ...)` - it doesn't hold a shell process.

Each wake-up costs an API call, but the prompt cache expires after 5 minutes of inactivity - balance accordingly."""
