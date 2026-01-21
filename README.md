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
git clone https://github.com/quantmew/ripperdoc.git
cd ripperdoc

# Install from source
pip install -e .
```



## Usage

### Interactive Mode (Recommended)
```bash
ripperdoc
# or use the short alias
rd
```

This launches an interactive session where you can:
- Ask questions about your codebase
- Request code modifications
- Execute commands
- Navigate and explore files

**Options:**
- `--yolo` - Skip permission prompts (safe mode is on by default)
- `--model <model_name>` - Specify a model (e.g., `claude-sonnet-4-20250514`, `gpt-4o`)
- `--tools <tool_list>` - Filter available tools (comma-separated, or "" for none)
- `--no-mcp` - Disable MCP server integration
- `--verbose` - Enable verbose logging

### Python SDK (headless)

Use Ripperdoc without the terminal UI via the new Python SDK. See [docs/SDK_USAGE.md](docs/SDK_USAGE.md) for examples of the one-shot `query` helper and the session-based `RipperdocClient`. 中文指南见 [docs/SDK_USAGE_CN.md](docs/SDK_USAGE_CN.md)。

#### SDK Examples

- **Basic Usage**: Simple one-shot queries
- **Session Management**: Persistent sessions with context
- **Tool Integration**: Direct tool access and customization
- **Configuration**: Custom model providers and settings

See the [examples/](examples/) directory for complete SDK usage examples.

### Safe Mode Permissions

Safe mode is enabled by default. When prompted:
- Type `y` or `yes` to allow a single operation
- Type `a` or `always` to allow all operations of that type for the session
- Type `n` or `no` to deny the operation

Use `--yolo` flag to skip all permission prompts:
```bash
ripperdoc --yolo
```

### Agent Skills

Extend Ripperdoc with reusable Skill bundles:

- **Personal skills**: `~/.ripperdoc/skills/<skill-name>/SKILL.md`
- **Project skills**: `.ripperdoc/skills/<skill-name>/SKILL.md` (can be checked into git)
- Each `SKILL.md` starts with YAML frontmatter:
  - `name` - Skill identifier
  - `description` - What the skill does
  - `allowed-tools` (optional) - Restrict which tools the skill can use
  - `model` (optional) - Suggest a specific model for this skill
  - `max-thinking-tokens` (optional) - Control thinking budget
  - `disable-model-invocation` (optional) - Use skill without calling the model
- Add supporting files alongside `SKILL.md` as needed
- Skills are auto-discovered and loaded on demand via the `Skill` tool

**Built-in skills:** PDF manipulation (`pdf`), PowerPoint (`pptx`), Excel (`xlsx`)

## Examples

### Code Analysis
```
> Can you explain what this function does?
> Find all references to the `parse_config` function
```

### File Operations
```
> Read the main.py file and suggest improvements
> Create a new component called UserProfile.tsx
> Update all imports to use the new package structure
```

### Code Generation
```
> Create a new Python script that implements a REST API client
> Generate unit tests for the auth module
> Add error handling to the database connection code
```

### Project Navigation
```
> Show me all the Python files in the project
> Find where the user authentication logic is implemented
> List all API endpoints in the project
```

### MCP Integration
```
> What MCP servers are available?
> Query the context7 documentation for React hooks
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
