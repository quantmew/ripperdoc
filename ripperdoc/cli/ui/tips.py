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
    "Press Alt+Enter to insert newlines in your input for multi-line prompts.",
    "Type / then press Tab to see all available slash commands.",
    "Double-press Ctrl+C to quickly exit Ripperdoc.",

    # Slash Commands
    "Use /help to see all available commands and get started.",
    "Use /clear to clear the current conversation and start fresh.",
    "Use /compact to compress conversation history when tokens run low.",
    "Use /status to check your session statistics and context usage.",
    "Use /cost to estimate token costs for your current session.",
    "Use /theme to switch between different visual themes (light, dark, etc.).",
    "Use /models to list and switch between different AI models.",
    "Use /tools to see what tools are available to the AI agent.",
    "Use /agents to manage specialized AI agents for specific tasks.",
    "Use /config to view or modify your Ripperdoc configuration.",
    "Use /todos to track tasks and manage your todo list.",

    # Advanced Features
    "Use /plan to enter planning mode for complex tasks requiring exploration.",
    "Use /memory to enable persistent memory across sessions.",
    "Use /skills to load and manage custom skill packages.",
    "Use /mcp to configure Model Context Protocol servers for extended capabilities.",
    "Use /permissions to adjust tool permission settings.",
    "Use /context to manage context window and conversation history.",
    "Use /resume to continue your most recent conversation.",

    # Productivity
    "Use /exit to cleanly shut down Ripperdoc and save your session.",
    "Use /doctor to diagnose common issues and configuration problems.",
    "Use /hooks to manage custom hooks for session lifecycle events.",

    # File Operations
    "Reference images with @path/to/image.png if your model supports vision.",
    "Use /tasks to view and manage background tasks.",

    # Hidden/Power User Features
    "Create custom commands in .ripperdoc/commands/ for reusable workflows.",
    "Configure hooks in .ripperdoc/hooks/ to automate session events.",
    "Use --yolo mode to skip all permission prompts (use with caution!).",
    "Pipe stdin into Ripperdoc: echo 'your query' | ripperdoc",
    "Use -p flag for single-shot queries: ripperdoc -p 'your prompt'",

    # MCP & Extensions
    "Connect to MCP servers to extend Ripperdoc with external tools and data sources.",
    "Use /skills to enable specialized capabilities like PDF or spreadsheet processing.",

    # Session Management
    "Sessions are automatically saved in .ripperdoc/sessions/.",
    "Use /continue or -c flag to resume your last conversation.",

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
