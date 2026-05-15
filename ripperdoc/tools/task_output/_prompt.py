"""TaskOutput tool prompt."""

TASK_OUTPUT_PROMPT = """- Retrieves output from a running or completed task
- Takes task_id, optional block (default true), and timeout (ms)
- Returns retrieval_status with task details
- timeout only limits waiting; it does not terminate the underlying task
- Use block=false for non-blocking checks
- Works with Bash background tasks and Task subagent runs"""
