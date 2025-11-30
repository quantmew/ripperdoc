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
- **MCP Server Support** - Connect to Model Context Protocol servers for extended capabilities
- **MCP Server Support** - Integration with Model Context Protocol servers
- **Subagent System** - Delegate tasks to specialized agents
- **Session Management** - Persistent session history and usage tracking
- **Jupyter Notebook Support** - Edit .ipynb files directly

## Available Tools

- **Bash** - Execute shell commands
- **BashOutput** - Read output from background commands
- **KillBash** - Terminate background commands
- **View** - Read file contents
- **Edit** - Edit files by replacing exact matches
- **MultiEdit** - Batch edit operations on files
- **NotebookEdit** - Edit Jupyter notebook cells
- **Write** - Create new files
- **Glob** - Find files matching patterns
- **Grep** - Search for patterns in files
- **LS** - List directory contents
- **TodoRead** - Read the current todo list or the next actionable task
- **TodoWrite** - Create and update persistent todo lists
- **Task** - Delegate work to specialized subagents
- **ListMcpServers** - List configured MCP servers and their tools
- **ListMcpResources** - List available resources from MCP servers
- **ReadMcpResource** - Read specific resources from MCP servers
- **Dynamic MCP Tools** - Runtime-loaded tools from connected MCP servers
- **Task** - Delegate tasks to specialized subagents
- **NotebookEdit** - Edit Jupyter notebook files
- **ListMcpServers** - List configured MCP servers
- **ListMcpResources** - List available MCP resources
- **ReadMcpResource** - Read specific MCP resources

## Project Structure

```
ripperdoc/
├── core/                    # Core functionality
│   ├── tool.py             # Base Tool interface
│   ├── query.py            # AI query system
│   ├── config.py           # Configuration management
│   ├── commands.py         # Command definitions
│   ├── permissions.py      # Permission system
│   ├── system_prompt.py    # System prompts
│   ├── agents.py           # Subagent management
│   └── default_tools.py    # Default tool configurations
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
│   ├── notebook_edit_tool.py
│   ├── task_tool.py
│   ├── mcp_tools.py
│   └── background_shell.py
├── utils/                  # Utility functions
│   ├── messages.py
│   ├── message_compaction.py
│   ├── log.py
│   ├── todo.py
│   ├── memory.py
│   ├── session_history.py
│   ├── session_usage.py
│   └── mcp.py
├── cli/                    # CLI interface
│   ├── cli.py             # Main CLI entry point
│   └── ui/                # UI components
│       ├── rich_ui.py     # Rich terminal UI
│       ├── context_display.py
│       └── spinner.py     # Loading spinners
│   └── commands/          # CLI commands
│       ├── agents_cmd.py
│       ├── config_cmd.py
│       ├── context_cmd.py
│       ├── cost_cmd.py
│       ├── help_cmd.py
│       ├── mcp_cmd.py
│       ├── models_cmd.py
│       ├── status_cmd.py
│       ├── tools_cmd.py
│       └── ...
└── sdk/                   # Python SDK
    └── client.py         # SDK client
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

#### SDK Examples

- **Basic Usage**: Simple one-shot queries
- **Session Management**: Persistent sessions with context
- **Tool Integration**: Direct tool access and customization
- **Configuration**: Custom model providers and settings

See the [examples/](examples/) directory for complete SDK usage examples.

### Safe Mode Permissions

Safe mode is the default. Use `--unsafe` to skip permission prompts. Choose `a`/`always` to allow a tool for the current session (not persisted across sessions).

### MCP Server Support

Ripperdoc supports Model Context Protocol (MCP) servers for extended functionality:

```bash
# List available MCP servers
ripperdoc mcp list

# List resources from a specific server
ripperdoc mcp resources <server-name>
```

Configure MCP servers in your configuration file or environment variables.

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
