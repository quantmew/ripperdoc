<div align="center">

# Ripperdoc

_an open-source, extensible AI coding agent that runs in your terminal_

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg">
  </a>
  <a href="https://github.com/quantmew/ripperdoc/stargazers">
    <img src="https://img.shields.io/github/stars/quantmew/ripperdoc.svg" alt="GitHub stars">
  </a>
</p>
</div>

Ripperdoc is your on-machine AI coding assistant, similar to [Claude Code](https://claude.com/claude-code), [Codex](https://github.com/openai/codex), [Gemini CLI](https://github.com/google-gemini/gemini-cli), [Aider](https://github.com/paul-gauthier/aider), and [Goose](https://github.com/block/goose). It can write code, refactor projects, execute shell commands, and manage files - all through natural language conversations in your terminal.

Designed for maximum flexibility, Ripperdoc works with **any LLM** (Anthropic Claude, OpenAI, DeepSeek, local models via OpenAI-compatible APIs), supports **custom hooks** to intercept and control tool execution, and offers both an interactive CLI and a **Python SDK** for headless automation.

[中文文档](README_CN.md) | [Contributing](CONTRIBUTING.md) | [Documentation](docs/)

## Features

- **AI-Powered Assistance** - Uses AI models to understand and respond to coding requests
- **Multi-Model Support** - Support for Anthropic Claude and OpenAI models
- **Rich UI** - Beautiful terminal interface with syntax highlighting
- **Code Editing** - Directly edit files with intelligent suggestions
- **Codebase Understanding** - Analyzes project structure and code relationships
- **Command Execution** - Run shell commands with real-time feedback
- **Tool System** - Extensible architecture with specialized tools
- **Agent Skills** - Load SKILL.md bundles to extend the agent on demand
- **Subagents** - Delegate tasks to specialized agents with their own tool scopes
- **File Operations** - Read, write, edit, search, and manage files
- **Todo Tracking** - Plan, read, and update persistent todo lists per project
- **Background Commands** - Run commands in background and monitor output
- **Permission System** - Safe mode with permission prompts for operations
- **Multi-Edit Support** - Batch edit operations on files
- **MCP Server Support** - Integration with Model Context Protocol servers
- **Session Management** - Persistent session history and usage tracking
- **Jupyter Notebook Support** - Edit .ipynb files directly
- **Hooks System** - Execute custom scripts at lifecycle events with decision control
- **Custom Commands** - Define reusable slash commands with parameter substitution

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

### Agent Skills

Extend Ripperdoc with reusable Skill bundles:

- Personal skills live in `~/.ripperdoc/skills/<skill-name>/SKILL.md`
- Project skills live in `.ripperdoc/skills/<skill-name>/SKILL.md` and can be checked into git
- Each `SKILL.md` starts with YAML frontmatter (`name`, `description`, optional `allowed-tools`, `model`, `max-thinking-tokens`, `disable-model-invocation`) followed by the instructions; add supporting files alongside it
- Model and max-thinking-token hints from skills are applied automatically for the rest of the session after you load them with the `Skill` tool
- Ripperdoc exposes skill names/descriptions in the system prompt and loads full content on demand via the `Skill` tool

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
