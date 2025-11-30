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

See DEVELOPMENT.md for detailed setup instructions.

Quick setup:
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
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

## Adding New Tools

To add a new tool:

1. Create a new file in `ripperdoc/tools/`
2. Extend the `Tool` base class
3. Implement all required methods
4. Add tests in `tests/test_tools.py`
5. Update `cli.py` to include the new tool
6. Add documentation

Example structure:
```python
from ripperdoc.core.tool import Tool, ToolResult
from pydantic import BaseModel

class MyToolInput(BaseModel):
    param: str

class MyToolOutput(BaseModel):
    result: str

class MyTool(Tool[MyToolInput, MyToolOutput]):
    @property
    def name(self) -> str:
        return "MyTool"

    async def description(self) -> str:
        return "Description for AI"

    # ... implement other methods
```

## Documentation

- Update README.md for user-facing changes
- Update DEVELOPMENT.md for developer-facing changes
- Add docstrings for all public APIs
- Include examples where appropriate

## Project Areas

Areas that need contributions:

### High Priority
- [ ] Interactive REPL mode
- [ ] More comprehensive tests
- [ ] OpenAI provider implementation
- [ ] Permission system for safe mode

### Medium Priority
- [ ] Cost tracking
- [ ] Session history
- [ ] More tools (ls, mkdir, etc.)
- [ ] Better error handling

### Low Priority
- [ ] MCP support
- [ ] Multi-model collaboration

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
