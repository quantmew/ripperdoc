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
  <a href="https://pypi.org/project/ripperdoc/">
    <img src="https://img.shields.io/badge/version-0.4.4-orange.svg">
  </a>
</p>

</div>

**Ripperdoc** is a powerful, extensible AI coding agent that runs directly in your terminal. Inspired by tools like [Claude Code](https://claude.com/claude-code), [Aider](https://github.com/paul-gauthier/aider), and [Goose](https://github.com/block/goose), Ripperdoc helps you write code, refactor projects, execute shell commands, and manage files through natural language conversations.

## What Makes Ripperdoc Different?

- **🔌 Model Agnostic** - Works with Anthropic Claude, OpenAI, Google Gemini, DeepSeek, and any OpenAI-compatible API
- **🎣 Extensible Architecture** - 30+ built-in tools with hooks system for custom workflows
- **🤖 Multi-Agent Coordination** - Built-in task graph and team collaboration for complex workflows
- **📚 Skill System** - Load capability bundles on-demand (PDF, Excel, PowerPoint, custom languages)
- **🔌 MCP Integration** - First-class Model Context Protocol server support
- **🛡️ Safe by Default** - Permission system with configurable rules and hooks
- **🎨 Beautiful UI** - Rich terminal interface with themes, syntax highlighting, and interactive TUIs
- **⚡ Background Tasks** - Run long-running commands asynchronously with real-time monitoring

[中文文档](README_CN.md) | [Contributing](CONTRIBUTING.md) | [Documentation](https://ripperdoc-docs.pages.dev/)

## Core Features

### 🛠️ Powerful Tool System
- **30+ Built-in Tools** - File operations (Read, Write, Edit, MultiEdit), code search (Grep, Glob), shell execution (Bash, Background), LSP integration, and more
- **Jupyter Support** - Direct .ipynb notebook editing with cell manipulation
- **Background Tasks** - Run commands asynchronously with output monitoring and status tracking

### 🤖 Multi-Agent Architecture
- **Task Graph System** - Persistent task management with dependencies, blockers, and ownership
- **Team Coordination** - Multi-agent collaboration with structured messaging and coordination
- **Specialized Subagents** - Built-in agents for code review, exploration, planning, and test generation

### 🔌 Extensibility
- **Skill System** - Load SKILL.md bundles to extend capabilities (PDF, Excel, PowerPoint, custom languages)
- **Hooks System** - Execute custom scripts at lifecycle events with decision control
- **Custom Commands** - Define reusable slash commands with parameter substitution
- **MCP Integration** - Connect to Model Context Protocol servers for extended capabilities

### 🎨 User Experience
- **Rich Terminal UI** - Beautiful interface with syntax highlighting and progress indicators
- **Theme Support** - Customizable color schemes and styling options
- **Interactive TUIs** - Terminal UIs for managing agents, models, permissions, and hooks
- **Safe Mode** - Permission prompts with configurable rules for dangerous operations

### 💾 Session Management
- **Persistent History** - Full conversation history with search and replay
- **Session Forking** - Create branches from any conversation state
- **Usage Tracking** - Monitor token usage and costs across sessions

## Installation

### Quick Install
```bash
pip install git+https://github.com/quantmew/ripperdoc.git
```

### From Source
```bash
git clone https://github.com/quantmew/ripperdoc.git
cd ripperdoc
pip install -e .
```

### Development Setup
```bash
# Install with development dependencies
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



## Quick Start

### Launch Interactive Session
```bash
ripperdoc
```

### Command-Line Options
```bash
ripperdoc [OPTIONS]
```

**Options:**
- `--yolo` - Skip permission prompts (safe mode is on by default)
- `--model <model_name>` - Specify model (e.g., `claude-sonnet-4-20250514`, `gpt-4o`)
- `--tools <tool_list>` - Filter available tools (comma-separated, or "" for none)
- `--no-mcp` - Disable MCP server integration
- `--verbose` - Enable verbose logging
- `--theme <theme_name>` - Set UI theme

**Environment Variables:**
- `RIPPERDOC_ENABLE_TASKS=false` - Use legacy Todo tools instead of Task Graph
- `RIPPERDOC_TASK_LIST_ID` - Force a shared Task Graph list ID across sessions
- `RIPPERDOC_MODEL` - Default model to use
- `RIPPERDOC_TEMPERATURE` - Default temperature (0.0-2.0)
- `RIPPERDOC_API_KEY` - API key for configured provider
- `RIPPERDOC_CONFIG_DIR` - Override where user-level Ripperdoc config/data are stored
- `RIPPERDOC_AUTOCOMPACT_PCT_OVERRIDE` - Override auto-compaction trigger percentage (1-100, capped at default threshold)
- `RIPPERDOC_TMPDIR` - Override internal temp root; Ripperdoc uses `<this-path>/ripperdoc/`
- `RIPPERDOC_EXIT_AFTER_STOP_DELAY` - In stdio/SDK mode, auto-exit after this many milliseconds of runtime idleness (positive integer only)
- `ENABLE_TOOL_SEARCH` - Controls deferred MCP tool search mode: `auto` (default, on at 10% context), `auto:N`, `true`, `false`

Task Graph scope behavior:
- By default, task lists are session-scoped (new session starts clean; resume/continue keeps the same tasks)
- Set `RIPPERDOC_TASK_LIST_ID` to intentionally share one task list across sessions

### Basic Usage Examples

**Code Analysis:**
```
> Can you explain what this function does?
> Find all references to the `parse_config` function
```

**File Operations:**
```
> Read the main.py file and suggest improvements
> Create a new component called UserProfile.tsx
> Update all imports to use the new package structure
```

**Code Generation:**
```
> Create a new Python script that implements a REST API client
> Generate unit tests for the auth module
> Add error handling to the database connection code
```

## Advanced Features

### Skills System

Extend Ripperdoc with reusable Skill bundles stored in `SKILL.md` files:

**Skill Locations:**
- `~/.ripperdoc/skills/<skill-name>/SKILL.md` (personal skills)
- `.ripperdoc/skills/<skill-name>/SKILL.md` (project-specific, can be committed to git)

**Skill Frontmatter:**
```yaml
---
name: pdf-processing
description: Comprehensive PDF manipulation toolkit
allowed-tools: Read, Write, Bash
model: claude-sonnet-4-20250514
max-thinking-tokens: 20000
---
```

**Built-in Skills:**
- `pdf` - PDF manipulation (extract text/tables, create, merge/split)
- `pptx` - PowerPoint presentation creation and editing
- `xlsx` - Excel spreadsheet operations with formulas
- `cangjie` - 仓颉 programming language support

### Hooks System

Execute custom scripts at lifecycle events with decision control:

**Hook Events:**
- `PreToolUse` - Before tool execution (can block/modify)
- `PostToolUse` - After successful tool execution
- `PostToolUseFailure` - After tool execution failure
- `PermissionRequest` - When permission is requested
- `UserPromptSubmit` - When user submits input
- `SessionStart/End` - Session lifecycle
- `SubagentStart/Stop` - Subagent lifecycle

**Hook Configuration:**
```json
{
  "hooks": [
    {
      "event": "PreToolUse",
      "command": "npm run lint",
      "blocking": true,
      "include_tools": ["Write", "Edit"]
    }
  ]
}
```

### MCP Integration

Ripperdoc supports the Model Context Protocol for extending capabilities:

```bash
# List available MCP servers
> /mcp

# Query MCP resources
> What MCP servers are available?
> Query the context7 documentation for React hooks
```

### Custom Commands

Define reusable slash commands with parameter substitution:

**Command File:** `.ripperdoc/commands/deploy.md`
```markdown
---
description: Deploy application to production
---

Deploying $ARGUMENTS to production...

!`npm run build && npm run deploy`
```

**Usage:** `/deploy my-feature-branch`

## Slash Commands

Ripperdoc provides powerful slash commands for session management:

**Session Commands:**
- `/exit` - Exit the session
- `/clear` - Clear conversation history
- `/compact` - Compact conversation history
- `/fork` - Create new session branch from current state
- `/resume` - Resume a previous session

**Configuration Commands:**
- `/config` - Manage configuration
- `/models` - Manage model providers
- `/tools` - View available tools
- `/permissions` - Manage permission rules
- `/hooks` - Manage hooks configuration
- `/themes` - Change UI theme
- `/output_language` - Set output language
- `/output_style` - Set output style

**Information Commands:**
- `/help` - Show help information
- `/status` - Show session status
- `/stats` - Show usage statistics
- `/cost` - Show cost tracking
- `/doctor` - Run system diagnostics

**Feature Commands:**
- `/skills` - List available skills
- `/agents` - Manage subagents
- `/tasks` - Task graph management
- `/todos` - Legacy todo management
- `/commands` - List custom commands
- `/context` - Manage context
- `/memory` - Manage memory
- `/mcp` - MCP server management

## Project Navigation

```
> Show me all the Python files in the project
> Find where the user authentication logic is implemented
> List all API endpoints in the project
> Explain the architecture of this codebase
```

## Background Tasks

```
> Run the tests in the background
> Start the dev server and monitor its output
> Check status of background tasks
```

## Architecture

### Core Components

- **CLI Layer** (`ripperdoc/cli/`) - Terminal interface, UI components, command handlers
- **Core Layer** (`ripperdoc/core/`) - Agent definitions, configuration, hooks, providers
- **Tools Layer** (`ripperdoc/tools/`) - 30+ built-in tools for file operations, code analysis, etc.
- **Protocol Layer** (`ripperdoc/protocol/`) - Stdio protocol handler for SDK communication
- **Utils Layer** (`ripperdoc/utils/`) - Shared utilities for logging, permissions, file operations

### Tool Categories

**File Operations:**
- `Read` - Read file contents with optional offset/limit
- `Write` - Create new files or overwrite existing
- `Edit` - Replace exact string matches in files
- `MultiEdit` - Batch edit operations on single file
- `NotebookEdit` - Edit Jupyter notebook cells

**Code Analysis:**
- `Grep` - Search code using regex patterns
- `Glob` - File pattern matching
- `LSP` - Language Server Protocol integration

**Shell Operations:**
- `Bash` - Execute shell commands
- `TaskStop` - Stop background tasks
- `TaskOutput` - Read output from background tasks

**Agent Features:**
- `TaskCreate/Update/Get/List` - Task graph management
- `TeamCreate/Delete` - Multi-agent team coordination
- `SendMessage` - Inter-agent messaging
- `Task` - Delegate to specialized subagents

**Extensibility:**
- `Skill` - Load skill bundles on-demand
- `ToolSearch` - Discover and activate tools
- `AskUserQuestion` - Interactive user prompts

### Supported Providers

- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus, Haiku
- **OpenAI** - GPT-4, GPT-4 Turbo, GPT-3.5
- **Google** - Gemini Pro, Gemini Flash
- **DeepSeek** - DeepSeek Coder, DeepSeek Chat
- **Custom** - Any OpenAI-compatible API

## Development

### Project Structure

```
ripperdoc/
├── cli/              # CLI interface and UI components
├── core/             # Core functionality and configuration
│   ├── hooks/        # Hooks system implementation
│   ├── providers/    # LLM provider implementations
│   └── query/        # Query processing loop
├── tools/            # Built-in tool implementations
├── protocol/         # Stdio protocol handler
│   └── stdio/        # Protocol implementation
├── utils/            # Utility functions
└── data/             # Model pricing and context data
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bash_tool.py

# Run with coverage
pytest --cov=ripperdoc
```

### Code Quality

```bash
# Type checking
mypy ripperdoc

# Code formatting
black ripperdoc

# Linting
ruff check ripperdoc

# Format with Black
black ripperdoc
```

## Configuration

### Config File Location
- `~/.ripperdoc/config.json` - User-level configuration
- `.ripperdoc/config.json` - Project-level configuration (overrides user config)

### Example Configuration

```json
{
  "model": "claude-sonnet-4-20250514",
  "temperature": 1.0,
  "api_key": "your-api-key",
  "permission_rules": {
    "Bash": {
      "rule": "ask",
      "commands": ["rm -rf", "sudo", ">:"]
    }
  },
  "hooks": {
    "hooks": [
      {
        "event": "PreToolUse",
        "command": "npm run lint",
        "blocking": true
      }
    ]
  },
  "theme": "default",
  "enable_tasks": true
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Key License Terms

- ✅ Commercial Use
- ✅ Distribution
- ✅ Modification
- ✅ Patent Grant
- ✅ Private Use
- ✅ Sublicensing
- ❌ Trademark Use

## Acknowledgments

Inspired by and built with ideas from:
- [Claude Code](https://claude.com/claude-code) - Anthropic's official CLI
- [aider](https://github.com/paul-gauthier/aider) - AI pair programming tool
- [Goose](https://github.com/block/goose) - Extensible AI assistant
- [Cursor](https://cursor.sh) - AI-powered code editor

## Resources

- [Documentation](https://ripperdoc-docs.pages.dev/)
- [中文文档](README_CN.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/quantmew/ripperdoc/issues)
- [Discussions](https://github.com/quantmew/ripperdoc/discussions)

---

**Made with ❤️ by the Ripperdoc team**
