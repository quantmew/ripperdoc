"""Random tips for Ripperdoc CLI.

This module provides a collection of tips and tricks for using Ripperdoc,
displayed randomly at startup to help users discover features.
"""

import random
from typing import List

# Tips database - organized by category
TIPS: List[str] = [
    # Navigation & Input
    "Press Tab to toggle thinking mode - see the AI's full reasoning process.",
    "Use @ followed by a file path to reference files in your conversation.",
    "Type @ and press Tab to autocomplete file mentions from your project.",
    "Press Alt+Enter to insert newlines in your input for multi-line prompts.",
    "Type / then press Tab to see all available slash commands.",
    "Press Ctrl+C once to clear the current input without exiting.",
    "Double-press Ctrl+C to quickly exit Ripperdoc.",
    "Double-press Esc to open history and roll back to a previous message.",

    # Slash Commands
    "Use /help to see all available commands and get started.",
    "Use /clear to clear the current conversation and start fresh.",
    "Use /compact to compress conversation history when tokens run low.",
    "Use /status to check your session statistics and context usage.",
    "Use /stats to see usage totals and a session heatmap.",
    "Use /cost to estimate token costs for your current session.",
    "Use /themes to switch between different visual themes (light, dark, etc.).",
    "Use /themes list or /themes preview <name> to browse themes.",
    "Use /models to list and switch between different AI models.",
    "Use /tools to see what tools are available to the AI agent.",
    "Use /agents to manage specialized AI agents for specific tasks.",
    "Use /config to view your current Ripperdoc configuration.",
    "Use /todos to track tasks and manage your todo list.",

    # Advanced Features
    "Use /memory to edit AGENTS.md for persistent preferences.",
    "Use /skills to list available skill packages.",
    "Use /mcp to list configured Model Context Protocol servers and tools.",
    "Use /permissions add/remove to tune tool permission prompts.",
    "Use /context to view context usage and token budget.",
    "Use /resume to continue your most recent conversation.",

    # Productivity
    "Use /exit to cleanly shut down Ripperdoc and save your session.",
    "Use /doctor to diagnose common issues and configuration problems.",
    "Use /hooks to manage custom hooks for session lifecycle events.",
    "Use /hooks add local to create an automation hook for this project.",

    # File Operations
    "Reference images with @path/to/image.png if your model supports vision.",
    "Use /tasks to view and manage background tasks.",
    "Use /tasks show <id> to view output from a background command.",
    "Use /tasks kill <id> to stop a running background command.",

    # Hidden/Power User Features
    "Create custom commands in .ripperdoc/commands/ for reusable workflows.",
    "Custom commands support $ARGUMENTS for simple parameterization.",
    "Configure hooks in .ripperdoc/hooks/ to automate session events.",
    "Use --yolo mode to skip all permission prompts (use with caution!).",
    "Use --cwd to run Ripperdoc in a different project directory.",
    "Pipe stdin into Ripperdoc: echo 'your query' | ripperdoc",
    "Use -p flag for single-shot queries: ripperdoc -p 'your prompt'",

    # MCP & Extensions
    "Connect to MCP servers to extend Ripperdoc with external tools and data sources.",
    "Project skills in .ripperdoc/skills override user skills with the same name.",

    # Session Management
    "Sessions are automatically saved in .ripperdoc/sessions/.",
    "Use /resume <id> to jump directly to a session by its id prefix.",
    "Use --continue or -c when launching Ripperdoc to resume your last conversation.",

    # Cost & Performance
    "Enable auto-compact in settings to automatically manage token usage.",
    "Use /cost to monitor token consumption and estimate API costs.",

    # Tips about Tips
    "A new tip appears each time you start Ripperdoc - try restarting to see more!",
]


def get_random_tip() -> str:
    """Get a random tip from the tips database.

    Returns:
        A randomly selected tip string.
    """
    return random.choice(TIPS)


def get_tips_count() -> int:
    """Get the total number of tips available.

    Returns:
        The count of tips in the database.
    """
    return len(TIPS)
