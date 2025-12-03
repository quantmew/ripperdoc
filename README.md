# Ripperdoc - AI-Powered Terminal Assistant

Ripperdoc is an AI-powered terminal assistant for coding tasks, providing an interactive interface for AI-assisted development, file management, and command execution.

[中文文档](README_CN.md) | [Contributing](CONTRIBUTING.md) | [Documentation](docs/)

## Features

- **AI-Powered Assistance** - Uses AI models to understand and respond to coding requests
- **Multi-Model Support** - Support for Anthropic Claude and OpenAI models
- **Rich UI** - Beautiful terminal interface with syntax highlighting
- **Code Editing** - Directly edit files with intelligent suggestions
- **Codebase Understanding** - Analyzes project structure and code relationships
- **Command Execution** - Run shell commands with real-time feedback
- **Tool System** - Extensible architecture with specialized tools
- **Subagents** - Delegate tasks to specialized agents with their own tool scopes
- **File Operations** - Read, write, edit, search, and manage files
- **Todo Tracking** - Plan, read, and update persistent todo lists per project
- **Background Commands** - Run commands in background and monitor output
- **Permission System** - Safe mode with permission prompts for operations
- **Multi-Edit Support** - Batch edit operations on files
- **MCP Server Support** - Integration with Model Context Protocol servers
- **Session Management** - Persistent session history and usage tracking
- **Jupyter Notebook Support** - Edit .ipynb files directly

## Installation

### Quick Installation
Install from git repository:
```bash
pip install git+https://github.com/quantmew/ripperdoc.git
```

Or install from source:
```bash
# Clone the repository
git clone <repository-url>
cd Ripperdoc

# Install from source
pip install -e .
```

### Configuration

Set your API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
# or for Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Interactive Mode (Recommended)
```bash
ripperdoc
```

This launches an interactive session where you can:
- Ask questions about your codebase
- Request code modifications
- Execute commands
- Navigate and explore files

### Python SDK (headless)

Use Ripperdoc without the terminal UI via the new Python SDK. See [docs/SDK_USAGE.md](docs/SDK_USAGE.md) for examples of the one-shot `query` helper and the session-based `RipperdocClient`. 中文指南见 [docs/SDK_USAGE_CN.md](docs/SDK_USAGE_CN.md)。

#### SDK Examples

- **Basic Usage**: Simple one-shot queries
- **Session Management**: Persistent sessions with context
- **Tool Integration**: Direct tool access and customization
- **Configuration**: Custom model providers and settings

See the [examples/](examples/) directory for complete SDK usage examples.

### Safe Mode Permissions

Safe mode is the default. Use `--unsafe` to skip permission prompts. Choose `a`/`always` to allow a tool for the current session (not persisted across sessions).

## Examples

### Code Analysis
```
> Can you explain what this function does?
```

### File Operations
```
> Read the main.py file and suggest improvements
```

### Code Generation
```
> Create a new Python script that implements a REST API client
```

### Project Navigation
```
> Show me all the Python files in the project
```

## Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy ripperdoc

# Code formatting
black ripperdoc

# Linting
ruff ripperdoc
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Key License Terms

- **Commercial Use**: Permitted
- **Distribution**: Permitted
- **Modification**: Permitted
- **Patent Grant**: Included
- **Private Use**: Permitted
- **Sublicensing**: Permitted
- **Trademark Use**: Not granted

For full license terms and conditions, please refer to the [LICENSE](LICENSE) file.

## Credits

Inspired by:
- [Claude Code](https://claude.com/claude-code) - Anthropic 官方 CLI
- [aider](https://github.com/paul-gauthier/aider) - AI pair programming tool
