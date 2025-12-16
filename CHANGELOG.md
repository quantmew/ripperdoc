# Changelog

All notable changes to Ripperdoc will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.5]: https://github.com/quantmew/Ripperdoc/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/quantmew/Ripperdoc/compare/v0.2.0...v0.2.4
[0.2.0]: https://github.com/quantmew/Ripperdoc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/quantmew/Ripperdoc/releases/tag/v0.1.0