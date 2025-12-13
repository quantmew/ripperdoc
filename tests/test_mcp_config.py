import pytest

from ripperdoc.utils import mcp


@pytest.mark.parametrize(
    "raw_command,raw_args,expected_cmd,expected_args",
    [
        ("npx -y @upstash/context7-mcp", None, "npx", ["-y", "@upstash/context7-mcp"]),
        ("python3", ["-m", "server"], "python3", ["-m", "server"]),
        (["node", "server.js"], None, "node", ["server.js"]),
        (None, None, None, []),
    ],
)
def test_normalize_command(raw_command, raw_args, expected_cmd, expected_args):
    cmd, args = mcp._normalize_command(raw_command, raw_args)
    assert cmd == expected_cmd
    assert args == expected_args
