# Ripperdoc - AI-Powered Terminal Assistant

Ripperdoc is an AI-powered terminal assistant for coding tasks, providing an interactive interface for AI-assisted development, file management, and command execution.

## Features

- **AI-Powered Assistance** - Uses AI models to understand and respond to coding requests
- **Multi-Model Support** - Support for Anthropic Claude and OpenAI models
- **Code Editing** - Directly edit files with intelligent suggestions
- **Codebase Understanding** - Analyzes project structure and code relationships
- **Command Execution** - Run shell commands with real-time feedback
- **Tool System** - Extensible architecture with specialized tools
- **Subagents** - Delegate tasks to specialized agents with their own tool scopes
- **Rich UI** - Beautiful terminal interface with syntax highlighting
- **File Operations** - Read, write, edit, search, and manage files
- **Todo Tracking** - Plan, read, and update persistent todo lists per project
- **Background Commands** - Run commands in background and monitor output
- **Permission System** - Safe mode with permission prompts for operations
- **Multi-Edit Support** - Batch edit operations on files

## Available Tools

- **Bash** - Execute shell commands
- **BashOutput** - Read output from background commands
- **KillBash** - Terminate background commands
- **View** - Read file contents
- **Edit** - Edit files by replacing exact matches
- **MultiEdit** - Batch edit operations on files
- **Write** - Create new files
- **Glob** - Find files matching patterns
- **Grep** - Search for patterns in files
- **LS** - List directory contents
- **TodoRead** - Read the current todo list or the next actionable task
- **TodoWrite** - Create and update persistent todo lists

## Project Structure

```
ripperdoc/
├── core/                    # Core functionality
│   ├── tool.py             # Base Tool interface
│   ├── query.py            # AI query system
│   ├── config.py           # Configuration management
│   ├── commands.py         # Command definitions
│   ├── permissions.py      # Permission system
│   └── system_prompt.py    # System prompts
├── tools/                  # Tool implementations
│   ├── bash_tool.py
│   ├── bash_output_tool.py
│   ├── kill_bash_tool.py
│   ├── file_edit_tool.py
│   ├── multi_edit_tool.py
│   ├── file_read_tool.py
│   ├── file_write_tool.py
│   ├── glob_tool.py
│   ├── grep_tool.py
│   ├── ls_tool.py
│   ├── todo_tool.py
│   └── background_shell.py
├── utils/                  # Utility functions
│   ├── messages.py
│   ├── message_compaction.py
│   ├── log.py
│   └── todo.py
└── cli/                    # CLI interface
    ├── cli.py             # Main CLI entry point
    └── ui/                # UI components
        ├── rich_ui.py     # Rich terminal UI
        ├── context_display.py
        └── spinner.py     # Loading spinners
```

## Installation

### Quick Installation
```bash
# Clone the repository
git clone <repository-url>
cd Ripperdoc

# Install from source
pip install -e .

# Or use the install script
./install.sh
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

### Quick Start

For a guided introduction, check out the [QUICKSTART.md](QUICKSTART.md) guide.

### Python SDK (headless)

Use Ripperdoc without the terminal UI via the new Python SDK. See [SDK_USAGE.md](SDK_USAGE.md) for examples of the one-shot `query` helper and the session-based `RipperdocClient`. 中文指南见 [SDK_USAGE_CN.md](SDK_USAGE_CN.md)。

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

### Project Documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [PYTERMGUI_USAGE.md](PYTERMGUI_USAGE.md) - PyTermGUI usage examples
- [CHANGELOG.md](CHANGELOG.md) - Release history
- [TODO.md](TODO.md) - Current development tasks

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details

## Credits

Inspired by:
- [Claude Code](https://claude.com/claude-code) - Anthropic 官方 CLI
- [aider](https://github.com/paul-gauthier/aider) - AI pair programming tool
