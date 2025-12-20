# Changelog

All notable changes to Ripperdoc will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.9]: https://github.com/quantmew/Ripperdoc/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/quantmew/Ripperdoc/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/quantmew/Ripperdoc/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/quantmew/Ripperdoc/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/quantmew/Ripperdoc/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/quantmew/Ripperdoc/compare/v0.2.0...v0.2.4
[0.2.0]: https://github.com/quantmew/Ripperdoc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/quantmew/Ripperdoc/releases/tag/v0.1.0
