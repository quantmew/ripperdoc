# Changelog

All notable changes to Ripperdoc will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
