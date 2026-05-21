"""Memory tool prompt."""

MEMORY_PROMPT = """Persistent memory file tool. Supports command=view/create/str_replace/insert/delete/rename over files in the session memory directory.

Input examples:
1. List current memory files
{
  "command": "view"
}

2. Create a topic memory file
{
  "command": "create",
  "path": "patterns.md",
  "content": "# Patterns\\n\\n- Prefer ripgrep for repository search.\\n"
}"""
