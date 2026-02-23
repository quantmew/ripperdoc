# Changelog

All notable changes to Ripperdoc will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] - 2026-02-23

### Added
- **Diff preview for edit permissions** - Edit and MultiEdit permission prompts now show a diff preview before applying changes
- **Adaptive onboarding palette** - Onboarding UI automatically adapts to light/dark terminal backgrounds using COLORFGBG detection
- **Theme-aware diff styles** - Choice UI diff styles now follow the active theme for consistent appearance
- **Restart notice for theme switching** - `/themes` command now displays a restart notice when switching themes

### Changed
- **EditResultRenderer improvement** - Now formats raw unified diff lines with line numbers instead of using preformatted tags
- **Tool count update** - Updated documentation to reflect 30+ built-in tools (was 26+)

## [0.4.3] - 2026-02-15

### Added
- **Conversation export command** - New `/export` command to export conversation history to clipboard or file
- **MCP server management TUI** - Interactive Textual-based UI for managing MCP servers with `/mcp` command
- **PyInstaller build system** - New build system with PyInstaller support including hooks and UPX compression
- **PyInstaller spec files** - Added spec files for both onefile and onedir build modes
- **Non-blocking MCP connections** - MCP server connections now use circuit breaker pattern for better resilience
- **Session usage tracking rebuild** - Rebuild session usage tracking with configurable replay message limit
- **Enhanced clear command** - `/clear` now resets session state completely with new `/reset` alias

### Changed
- **Model catalog performance** - Migrated model catalog from JSON to Python module for faster loading
- **Build system migration** - Migrated from Nuitka to PyInstaller for better cross-platform binary support
- **Provider error handling** - Standardized error handling across all providers with unified error types
- **Permissions TUI** - Improved exit behavior via Escape key in permissions management TUI
- **File tool display** - Removed `file_path:` prefix from tool argument display for cleaner output
- **Provider configurations** - Updated provider configurations and cleaned up model data

### Fixed
- **Windows executable naming** - Handle Windows executable naming correctly in build script
- **OpenAI error handling** - Enhanced error handling for httpx transport exceptions in OpenAI provider
- **Build scripts** - Various improvements to PyInstaller and Nuitka build scripts

### Removed
- **UPX auto-download** - Removed automatic UPX download from build script for security

## [0.4.2] - 2026-02-10

### Added
- **Hide completed tasks** - Completed tasks are now hidden from UI output by default for cleaner display
- **Display completed tasks option** - New opt-in option to show completed tasks when needed
- **MCP stderr log inspection** - New `/mcp logs` command for inspecting MCP server stderr logs
- **Thinking mode configuration** - Support for thinking mode configuration in model profiles
- **Enhanced model configuration** - Separate max input/output tokens configuration for models

### Fixed
- **Flaky async hook test** - Fixed flaky async hook test by increasing wait time
- **Anthropic extended thinking signatures** - Properly handle Anthropic extended thinking signatures in streaming and message replay
- **Token management** - Remove redundant context_window field and simplify token management

### Changed
- **MCP runtime initialization** - Improved type safety and optimized MCP runtime initialization
- **Plugin uninstall confirmation** - Added confirmation dialog for plugin uninstall operations
- **Documentation sync** - Synced Chinese README with English version
- **CLI alias removal** - Removed `rd` CLI alias for consistency

### Refactored
- **Task ID display** - Removed "id:" prefix from task ID display for cleaner output
- **Task ID separator** - Changed task ID separator from hyphen to colon for better display consistency

## [0.4.1] - 2026-02-08

### Added
- **Plugin system with marketplace support** - Comprehensive plugin architecture enabling users to extend Ripperdoc with custom commands, skills, agents, hooks, and MCP/LSP servers
- **Plugin marketplace operations** - Support for discovering, installing, and managing plugins from online marketplaces
- **Plugin scope management** - User, project, and local scopes for flexible plugin installation and usage
- **Plugin registry system** - Centralized registry for tracking installed plugins across different scopes
- **Plugin Textual TUI** - Interactive terminal UI for comprehensive plugin management

### Fixed
- **Context budget validation** - Add ContextBudgetConfigurationError for invalid model token configurations
- **MCP server resilience** - Isolate MCP server connection failures to prevent cascade errors
- **Token budget calculation** - Support split input/output token budgets without double subtraction
- **Task ID display formatting** - Simplify task ID display format for cleaner output

### Changed
- **Plugin system enhancements** - Improved marketplace operations with better error handling and feedback
- **Task rendering optimization** - Streamlined task ID display format

## [0.4.0] - 2026-02-08

### ⚠️ BREAKING CHANGES - Configuration File Format Incompatible

**IMPORTANT: Please backup your configuration files before upgrading!**

This release includes major architectural changes that introduce **incompatible changes to the configuration file format**. Before upgrading to v0.4.0, please:

1. **Backup your configuration**: Copy your `~/.ripperdoc/c`, `~/.ripperdoc.json`, and any project-specific `.ripperdoc/` files
2. **Review provider settings**: The `provider` field has been renamed to `protocol`
3. **Update model configurations**: Model profiles now use a new structure with auto-completion from the built-in model catalog

### Added
- **Task graph and team collaboration system** - Complete task management with persistent task graph, dependency tracking, and multi-agent team coordination
- **Model catalog system** - Comprehensive built-in model catalog with pricing, context windows, and capability metadata for 200+ models
- **Output style customization** - New `/output-style` command with 8+ predefined output styles (concise, detailed, json, markdown, etc.)
- **Output language control** - New `/output-language` command to control AI response language independently from UI language
- **Session-scoped working directories** - Support for multiple working directories per session with `/add-dir` command
- **Interactive question system** - New `AskUserQuestion` tool for single and multi-select interactive prompts
- **Custom commands management TUI** - Interactive terminal UI for managing custom commands with `/commands` slash command
- **Skills enable/disable** - New TUI and commands to enable/disable individual skills
- **Enhanced bash output** - Detailed truncation metadata for command output with size and line limit information
- **SDK can_use_tool integration** - Permission preview support for SDK tool usage
- **Async hook support** - Hooks now support async commands and bidirectional SDK protocol callbacks
- **Hook scopes** - Add hook scopes for skills and agents for more granular control
- **Agent hook type** - New hook event type specifically for agent lifecycle events
- **JSON output format** - New JSON output mode for programmatic consumption
- **Auto input format** - Enhanced input parsing with automatic format detection
- **Custom system prompts** - Support for per-query custom system prompts
- **Max turns option** - Limit conversation turns with `--max-turns` option
- **Sidecar session index** - Fast session lookup and statistics with new index file format
- **Python 3.10 asyncio compatibility** - Add asyncio.timeout compatibility layer for Python 3.10

### Changed
- **Provider → Protocol rename** - Renamed `ProviderType` to `ProtocolType` throughout codebase for clarity
- **Core module reorganization** - Major refactoring of core modules:
  - `query_utils.py` → `message_utils.py`
  - `permissions.py` → `permission_engine.py`
  - `provider_catalog.py` → `provider_metadata.py`
  - `default_tools.py` → `tool_defaults.py`
- **Model profile structure** - Model profiles now auto-populate from model catalog with pricing, context windows, and capabilities
- **Bash output rendering** - Simplified bash output rendering with improved test coverage
- **Query runtime helpers** - Extracted query runtime helpers for better modularity
- **Message normalization** - Enhanced message normalization in query loop
- **Permission decision engine** - Modularized permission decision engine into separate functions
- **Dynamic subagent prompts** - Subagent prompts now adapt based on available tools
- **Environment variable handling** - Simplified environment variable handling across the codebase
- **Stdio handler** - Split stdio handler into mixins for better separation of concerns
- **Session resume replay** - Improved tool rendering in conversation replay for session resume
- **SessionStart/End hooks** - Ensure hooks execute once per session with deterministic cleanup
- **Permission mode synchronization** - Sync permission mode across query context, hooks, and permissions
- **Thinking spinner** - Auto-refresh thinking spinner via async tick loop
- **CLI integration** - Integrated stdio mode into main CLI with SDK-compatible options
- **TUI notifications** - Replace inline error widgets with app notifications in TUI forms
- **Internationalization** - Translated Chinese comments to English in source code
- **Public API promotion** - Promoted internal utilities to public API for extensibility
- **Type annotation coverage** - Improved type annotation coverage and mypy compatibility

### Fixed
- **Model profile pricing** - Use model profile pricing and fix num_turns calculation
- **Permission override** - Ensure deny/block decisions override allow in hooks
- **Gitignore warnings** - Suppress gitignore warnings when writing files
- **Input validation** - Add input validation tests and edge case coverage

### Removed
- **Outdated examples** - Removed outdated example scripts and documentation
- **Unused commands module** - Removed unused core commands module
- **Legacy provider aliases** - Removed legacy provider string aliases (use protocol types instead)

## [0.3.3] - 2026-02-04

### Added
- **Conversation history picker** - Add interactive conversation history picker accessible via double-Esc shortcut for quick access to previous messages
- **Session fork and resume** - Add `/fork` command to create new session branches from current conversation state and resume capability
- **Interactive hooks management TUI** - Add comprehensive terminal UI for managing hooks with visual interface
- **New hook events** - Add new hook events for enhanced extensibility
- **ESC key to quit TUIs** - Add ESC key binding to quit agents and models management TUIs
- **OLDPWD validation** - Add validation support for `cd -` command to prevent directory traversal issues
- **Enhanced user tips** - Add new commands and improved descriptions to random tips system

### Changed
- **Default temperature** - Update default model temperature from 0.7 to 1.0 for more creative responses
- **Hooks system improvements** - Improve hook system architecture with manager and better integration

### Fixed
- **Double exception during tool cancellation** - Avoid raising double exceptions during concurrent tool cancellation
- **Grep output parsing** - Improve output parsing for paths with colons and Windows paths
- **Gitignore matcher** - Implement directory-aware gitignore matcher with proper pattern scoping
- **Rich UI modularization** - Split rich_ui.py into modular components for better maintainability

### Refactored
- **Query module** - Split query module into package structure (context, errors, loop, permissions, tools)
- **Protocol module** - Reorganize stdio protocol into package structure (command, handler, timeouts, watchdog)
- **Git utilities** - Implement directory-aware gitignore matcher with proper pattern scoping
- **Permissions system** - Refactor permission interpreter and add shell command validation

## [0.3.2] - 2026-02-03

### Added
- **Ask rules support** - Add explicit ask rules to permission system for commands requiring user confirmation
- **Interactive management TUIs** - Add interactive terminal UIs for managing agents, models, and permissions
- **Cache tokens tracking** - Add support for Anthropic cache tokens and alternative API usage tracking
- **Random tips system** - Add random tips feature to help users discover features during usage
- **Tool filtering** - Add tool filtering capability and optimize file slice reading performance

### Changed
- **Unified choice component** - Introduce unified choice component and refactor permission prompts for consistency
- **Session selection UI** - Modernize session selection interface with interactive choice component
- **Theme selection UI** - Enhance theme selection with custom styling and interactive prompts
- **Welcome panel and status bar** - Improve UX information display in welcome panel and status bar
- **File editing utilities** - Extract common file editing utilities to shared module for better code organization

### Fixed
- **Long tool descriptions** - Truncate long tool descriptions in MCP server list for better display
- **Terminal line calculation** - Improve terminal line count calculation for choice prompts
- **Ask rules precedence** - Ensure proper precedence: deny > ask > allow in permission evaluation

## [0.3.1] - 2026-02-02

### Added
- **ESC key for permission denial** - Press ESC to quickly deny permission requests
- **ESC key interrupt support** - Press ESC to interrupt running queries

### Changed
- **Permission prompts refactor** - Refactor permission prompts using prompt_toolkit for better consistency
- **Permission prompt styling** - Optimize the style and color of permission check prompts for better visual feedback

### Fixed
- **ESC interrupt handling** - Remove ESC interrupt functionality during query execution to avoid agent termination issues

## [0.3.0] - 2026-01-31

### Added
- **Theme management system** - Comprehensive theme support with custom color schemes and styling options
- **Image input support** - Add support for image input in queries with automatic image processing
- **Sub-process SDK architecture** - Complete rewrite of SDK with sub-process architecture for better isolation and reliability
- **Claude Agent SDK compatibility** - Full compatibility with Claude Agent SDK protocol and interfaces
- **File encoding detection** - Automatic file encoding detection and handling for international text files
- **Tool execution timeout** - Implement timeout mechanism for tool execution to prevent hanging
- **Environment variable configuration** - Support `RIPPERDOC_` prefixed environment variables for model parameters
- **Detailed tool call logging** - Enhanced logging system with detailed tool call information
- **Thinking mode toggle** - Add toggle and status display for thinking mode
- **Skill list command** - New `/skills` command to list available skills
- **Stdin pipe input** - Support piped stdin input for initial queries
- **Session continuation** - Implement session continuation feature with enhanced logging
- **Pending message queue** - Implement pending message queue for background tasks
- **Fuzzy command matching** - Add fuzzy matching for mistyped slash commands with suggestions
- **Background task tracking** - Add runtime and age tracking for background tasks
- **Platform utilities** - Unified platform detection logic and platform utility module
- **LSP and Skill tools** - Add dedicated LSP and Skill tools with enhanced functionality
- **Sub-agent message forwarding** - Add message forwarding functionality for sub-agents
- **File read limits** - Enhanced file reading tool with size and line limits
- **Doctor command** - Add `/doctor` command for system diagnostics

### Changed
- **Protocol package refactor** - Move stdio_cmd module to protocol package for better organization
- **SDK removal** - Remove legacy in-process SDK implementation
- **Query processing improvements** - Ensure tool inputs are always dictionaries with better validation
- **Parallel tool execution** - Improved error handling for parallel tool execution with tool names in context
- **File operations enhancement** - Enhanced file editing, reading, and writing tools with encoding support
- **Background shell improvements** - Better integration with pending message queue
- **UI improvements** - Enhanced rich UI with better rendering and user interaction
- **Spinner fixes** - Fix terminal output flickering and layout issues
- **Permission checker** - Improved tooltip handling and rendering
- **Error messages** - Enhanced error messages with model information in error and interrupt messages
- **Git utilities** - New comprehensive git utility functions
- **Shell utilities** - Improved shell path handling using PureWindowsPath
- **Message formatting** - Enhanced message formatting with better image support

### Fixed
- **Timestamp conversion** - Correct timestamp conversion to use UTC in StructuredFormatter
- **Interrupt handling** - Refine interrupt key detection logic on Windows
- **Tool input validation** - Ensure tool inputs are always dictionaries
- **File mention completer** - Improved file mention completion with better search
- **Type checking** - Format code and fix type-checking issues
- **Test coverage** - Add comprehensive test coverage for new features

### Removed
- **Legacy SDK docs** - Remove old SDK documentation
- **QUICKSTART.md** - Remove quickstart guide (integrated into main README)
- **In-process SDK mode** - Remove in-process mode in favor of sub-process architecture

## [0.2.10] - 2026-01-14

### Added
- **LSP tool integration** - Add Language Server Protocol support for code intelligence queries (goToDefinition, findReferences, hover, etc.)
- **Session statistics and heatmap** - Visualize session activity patterns with statistics panel and heatmap visualization
- **Bounded file cache** - Implement memory-managed file snapshot caching with size limits
- **Enhanced CLI options** - Add `--tools` filter for limiting available tools, custom system prompts, and model selection
- **Double Ctrl+C exit** - Implement improved exit handling with double Ctrl+C functionality
- **Quickstart guide** - Add comprehensive quickstart documentation for new users
- **Background agent execution** - Support for running agents in background mode
- **Flexible permission patterns** - Enhanced permission system with wildcard matching and glob patterns
- **New keyboard shortcuts** - Additional keyboard shortcuts for improved user interaction

### Changed
- **UI rendering improvements** - Replace Rich Status with Live+Spinner for cleaner terminal output
- **Permission system refactor** - Update shell command validation to use permission checks
- **Model pointer simplification** - Reduce model pointer system from 4 to 2 pointers for better performance
- **Session statistics optimization** - Consolidate longest session and activity patterns calculation
- **Agent management enhancements** - Improved session tracking and hooks integration
- **Permission rule integration** - Better global/local rule management in permission system

### Fixed
- **Assistant task cancellation** - Prevent unnecessary cancellation of assistant_task during query iteration
- **Bounded file cache memory** - Ensure memory size does not go negative and improve reasoning preview logic
- **Error handling improvements** - Enhanced validation in tools and better memory management
- **Code quality issues** - Resolve ruff and mypy code quality problems
- **Legacy code cleanup** - Remove outdated code and optimize query queue handling
- **LSP error handling** - Improve error handling for Language Server Protocol operations

## [0.2.9] - 2025-12-20

### Added
- **Full thinking output flag** - New `--show-full-thinking` CLI option to display full provider reasoning
- **Spinner pause control** - UI spinner now pauses around blocking operations for cleaner terminal output

### Changed
- **Permission mode rename** - Renamed `safe_mode` to `yolo_mode` to better describe relaxed permission behavior
- **Provider selection refactor** - Streamlined provider selection logic and defaults
- **Shell command validation** - Improved quote handling and interpreter detection for safer execution
- **Hooks typing improvements** - Better type annotations and parameter handling for hooks

### Fixed
- **DeepSeek tool call compatibility** - Include `reasoning_content` for DeepSeek API tool calls to prevent errors

## [0.2.8] - 2025-12-19

### Added
- **Comprehensive hooks system** - Custom script execution at various points during operation (PreToolUse, PermissionRequest, PostToolUse, etc.)
- **Custom command system** - File-based command definitions with frontmatter support, parameter substitution, and bash command execution
- **Enhanced hooks CLI** - Guided add/edit/delete operations for hook management
- **Hook examples** - Comprehensive example scripts for various use cases (logging, validation, notifications, etc.)

### Changed
- **Improved hooks integration** - Better integration with existing permission and tool systems
- **Enhanced UI for hooks management** - User-friendly interface for managing hooks configuration

### Fixed
- **Type checking and code quality issues** - Various fixes for better code reliability

## [0.2.7] - 2025-12-19

### Added
- **Message formatting utilities** - Improved conversation compaction and message handling

### Changed
- **Enhanced UI components** - Better modularization and user interface improvements
- **Improved type annotations** - Better code quality and type checking

### Fixed
- **Type checking issues** - Resolved various type checking and code quality problems

## [0.2.6] - 2025-12-18

### Added
- **@ symbol file mention completion** - Type `@` followed by Tab to autocomplete file paths with recursive search
- **ESC key interrupt support** - Press ESC to interrupt running queries and commands
- **UI modularization** - New modular UI components for better maintainability
- **Conversation compaction** - Automatic conversation length management to handle context limits
- **Session resume pagination** - Navigate through session history with paginated interface
- **Improved command aliases** - Enhanced command shortcuts and usability

### Changed
- **UI refactoring** - Major UI codebase reorganization into modular components
- **Enhanced interrupt handling** - Simplified ESC/Ctrl+C interrupt handling with single raw-mode loop
- **Improved permission dialogs** - Better handling of ESC listener during permission checks
- **Better conversation management** - Enhanced message compaction and conversation state handling

### Fixed
- **ESC key blocking** - Fixed ESC key listener blocking terminal during permission dialogs
- **Pydantic validation** - Fixed file_state_cache validation for proper cache sharing
- **Shell command permissions** - Improved user confirmation for sensitive directory operations

## [0.2.5] - 2025-12-17

### Added
- Enhanced shell command permissions with user confirmation for sensitive directories
- Comprehensive path ignore system with gitignore-style patterns
- Improved shell command security with destructive command detection
- Comprehensive permission management system
- File state validation to prevent race conditions
- Context-length error detection for auto-compaction support
- Enhanced skill system with context updates and validation
- Longer MCP lifecycle support
- Truncation for long grep results
- Thinking max tokens configuration

### Changed
- Improved error handling and query loop architecture
- Enhanced tool rendering and model management
- Better type annotations, error handling, and code formatting
- Improved system prompt with better guidance and tool integration

## [0.2.4] - 2025-12-16

### Added
- Experimental support for skills.md
- Explore and plan agents with improved MCP command parsing

### Changed
- Improved type annotations and error handling across providers
- Enhanced system prompt with improved guidance and tool integration

## [0.2.0] - 2025-12-15

### Added
- Python SDK for headless usage
- MCP (Model Context Protocol) server integration
- Skill system for extending agent capabilities
- Subagent system for task delegation
- Comprehensive tool system with 20+ built-in tools
- Multi-model support (Anthropic, OpenAI-compatible, Gemini)
- Permission system with safe mode as default
- Session management and history tracking
- Jupyter Notebook support
- Todo tracking system

### Changed
- Complete rewrite of the architecture for better extensibility
- Improved security with command validation and path restrictions
- Enhanced UI with rich terminal interface

## [0.1.0] - 2025-11-01

### Added
- Initial release
- Basic AI-powered terminal assistant
- File operations and code editing
- Simple command execution
- Basic project navigation

[0.4.3]: https://github.com/quantmew/Ripperdoc/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/quantmew/Ripperdoc/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/quantmew/Ripperdoc/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/quantmew/Ripperdoc/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/quantmew/Ripperdoc/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/quantmew/Ripperdoc/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/quantmew/Ripperdoc/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/quantmew/Ripperdoc/compare/v0.2.10...v0.3.0
[0.2.10]: https://github.com/quantmew/Ripperdoc/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/quantmew/Ripperdoc/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/quantmew/Ripperdoc/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/quantmew/Ripperdoc/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/quantmew/Ripperdoc/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/quantmew/Ripperdoc/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/quantmew/Ripperdoc/compare/v0.2.0...v0.2.4
[0.2.0]: https://github.com/quantmew/Ripperdoc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/quantmew/Ripperdoc/releases/tag/v0.1.0
