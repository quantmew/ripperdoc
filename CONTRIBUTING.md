# Contributing to Ripperdoc

Thank you for your interest in contributing to Ripperdoc! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Describe the issue clearly
- Include steps to reproduce
- Mention your environment (OS, Python version, etc.)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

Quick setup:
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public functions
- Keep lines under 100 characters

We use:
- `black` for code formatting
- `mypy` for type checking
- `ruff` for linting

Run before committing:
```bash
black ripperdoc
mypy ripperdoc
ruff check ripperdoc
```

## Testing

All new features should include tests:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tools.py

# Run with coverage
pytest --cov=ripperdoc
```

## Project Architecture

### Core Modules

- `ripperdoc/core/` - Core functionality (config, agents, permissions, providers)
- `ripperdoc/tools/` - Built-in tools (26+ tools for file ops, shell, search, etc.)
- `ripperdoc/cli/` - CLI commands and TUI interfaces
- `ripperdoc/protocol/` - SDK protocol implementation
- `ripperdoc/utils/` - Utility functions and helpers

### Key Features

- **Multi-Provider Support** - Anthropic, OpenAI, Gemini, and OpenAI-compatible APIs
- **30+ Built-in Tools** - File operations, shell execution, code search, LSP integration
- **Task Graph System** - Persistent task management with dependencies
- **Team Collaboration** - Multi-agent coordination with structured messaging
- **Plugin System** - Extensible via plugins, skills, hooks, and MCP servers
- **Permission System** - Configurable rules for safe operation

## Adding New Tools

To add a new tool:

1. Create a new file in `ripperdoc/tools/`
2. Extend the `Tool` base class from `ripperdoc.core.tool`
3. Implement required methods: `name`, `description`, `input_schema`, `run`
4. Register the tool in `ripperdoc/core/tool_defaults.py`
5. Add tests in `tests/`
6. Update documentation

Example structure:
```python
from ripperdoc.core.tool import Tool, ToolResult
from pydantic import BaseModel

class MyToolInput(BaseModel):
    param: str

class MyTool(Tool[MyToolInput, ToolResult]):
    @property
    def name(self) -> str:
        return "MyTool"

    async def description(self) -> str:
        return "Description for AI"

    def input_schema(self) -> type[MyToolInput]:
        return MyToolInput

    async def run(self, input_data: MyToolInput, context: ToolUseContext) -> ToolResult:
        # Implementation here
        return ToolResult(data={"result": "success"})
```

## Adding New Providers

To add support for a new AI provider:

1. Create a new file in `ripperdoc/core/providers/`
2. Extend the `BaseProvider` class
3. Implement required methods for your provider
4. Add provider metadata in `ripperdoc/core/provider_metadata.py`
5. Add tests in `tests/`

## Adding New CLI Commands

To add a new slash command:

1. Create a new file in `ripperdoc/cli/commands/`
2. Extend the `BaseCommand` class from `ripperdoc/cli/commands/base.py`
3. Register the command in `ripperdoc/cli/cli.py`
4. Add tests in `tests/`

## Adding Skills

Skills are capability bundles defined in `SKILL.md` files. To add a new skill:

1. Create a directory in `skills/` with a `SKILL.md` file
2. Define the skill's purpose, tools, and instructions
3. Add any supporting files (examples, templates)
4. Update documentation

## Adding Hooks

Hooks allow custom scripts to run at lifecycle events. Hook types include:

- `PreToolUse` - Before tool execution
- `PostToolUse` - After tool execution
- `PermissionRequest` - During permission prompts
- `SessionStart` / `SessionEnd` - Session lifecycle

## Documentation

- Update README.md for user-facing changes
- Update README_CN.md for Chinese documentation sync
- Add docstrings for all public APIs
- Include examples where appropriate
- Update CHANGELOG.md with your changes

## Project Areas for Contribution

### High Priority
- [ ] Improved error recovery and retry mechanisms

### Medium Priority
- [ ] Command auto-completion
- [ ] Session tagging and search
- [ ] Configurable keyboard shortcuts
- [ ] Database connection tools

### Low Priority
- [ ] Startup speed optimization
- [ ] Memory usage optimization
- [ ] Structured logging system

## Release Process

1. Update version in `ripperdoc/__init__.py`
2. Update CHANGELOG.md with release notes
3. Create a git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. GitHub Actions will build and publish to PyPI

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
