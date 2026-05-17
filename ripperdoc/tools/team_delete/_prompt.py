"""Prompt for TeamDeleteTool."""

from __future__ import annotations

from textwrap import dedent

TEAM_DELETE_PROMPT = dedent(
    """\
    # TeamDelete

    Remove team and task directories when the swarm work is complete.

    This operation:
    - Removes the team directory (`~/.ripperdoc/teams/{team-name}/`)
    - Removes the task directory (`~/.ripperdoc/tasks/{team-name}/`)
    - Clears team context from the current session

    **IMPORTANT**: TeamDelete will fail if the team still has active members. Gracefully terminate teammates first, then call TeamDelete after all teammates have shut down.

    Use this when all teammates have finished their work and you want to clean up the team resources. The team name is automatically determined from the current session's team context.
    """
).strip()
