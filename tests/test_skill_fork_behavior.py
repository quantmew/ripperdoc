from ripperdoc.core.query.loop import _collect_skill_fork_requests
from ripperdoc.utils.messaging.messages import create_user_message


def test_collect_skill_fork_requests_builds_task_payload() -> None:
    conversation = [create_user_message("Please fix this bug and add tests.")]
    skill_result = create_user_message(
        "tool result",
        tool_use_result={
            "skill": "bug-fix",
            "context": "fork",
            "agent": "general-purpose",
            "model": "quick",
            "paths": ["src/**", "tests"],
            "content": "Investigate failing tests and patch the implementation.",
        },
    )

    requests = _collect_skill_fork_requests([skill_result], conversation)

    assert len(requests) == 1
    request = requests[0]
    assert request.skill_name == "bug-fix"
    assert request.subagent_type == "general-purpose"
    assert request.model == "quick"
    assert "Please fix this bug and add tests." in request.prompt
    assert "Investigate failing tests" in request.prompt
    assert "Preferred path scope" in request.prompt


def test_collect_skill_fork_requests_skips_missing_agent() -> None:
    conversation = [create_user_message("Do work")]
    skill_result = create_user_message(
        "tool result",
        tool_use_result={
            "skill": "no-agent",
            "context": "fork",
            "content": "Instructions",
        },
    )

    requests = _collect_skill_fork_requests([skill_result], conversation)
    assert requests == []
