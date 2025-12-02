"""Example configuration for Ripperdoc.

This file shows how to configure Ripperdoc for different use cases.
"""

# Example 1: Basic Anthropic Configuration
BASIC_ANTHROPIC = {
    "model_profiles": {
        "default": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": "your_anthropic_api_key_here",
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 200000,
        }
    },
    "model_pointers": {
        "main": "default",
        "task": "default",
        "reasoning": "default",
        "quick": "default",
    },
    "theme": "dark",
    "verbose": False,
    "safe_mode": False,
}

# Example 2: Multi-Model Configuration
MULTI_MODEL = {
    "model_profiles": {
        "sonnet": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": "your_anthropic_api_key",
            "max_tokens": 8192,
            "temperature": 0.7,
            "context_window": 200000,
        },
        "haiku": {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "api_key": "your_anthropic_api_key",
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 200000,
        },
        "gpt4": {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "api_key": "your_openai_api_key",
            "max_tokens": 4096,
            "temperature": 0.7,
            "context_window": 128000,
        },
    },
    "model_pointers": {
        "main": "sonnet",  # Use Sonnet for main tasks
        "task": "haiku",  # Use Haiku for sub-tasks
        "reasoning": "sonnet",  # Use Sonnet for reasoning
        "quick": "haiku",  # Use Haiku for quick tasks
    },
    "theme": "dark",
    "verbose": True,
}

# Example 3: Project-Specific Configuration
PROJECT_CONFIG = {
    "allowed_tools": ["Bash", "View", "Edit", "Glob", "Grep"],
    "context": {
        "project_type": "Python Web Application",
        "framework": "FastAPI",
        "database": "PostgreSQL",
    },
    "context_files": ["README.md", "requirements.txt", "src/main.py"],
    "dont_crawl_directory": False,
    "enable_architect_tool": False,
}

# Example 4: Safe Mode Configuration
SAFE_MODE = {
    "model_profiles": {
        "default": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": "your_api_key",
        }
    },
    "safe_mode": True,  # Always ask for permission
    "verbose": True,  # Show all operations
}


def create_config_file(config: dict, path: str = "~/.ripperdoc.json"):
    """
    Create a configuration file from a config dictionary.

    Usage:
        from ripperdoc.examples.config_examples import MULTI_MODEL, create_config_file
        create_config_file(MULTI_MODEL, "~/.ripperdoc.json")
    """
    import json
    from pathlib import Path

    config_path = Path(path).expanduser()
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Configuration written to {config_path}")


if __name__ == "__main__":
    print("Example Ripperdoc Configurations")
    print("=" * 50)
    print("\n1. Basic Anthropic Configuration")
    print("2. Multi-Model Configuration")
    print("3. Project-Specific Configuration")
    print("4. Safe Mode Configuration")
    print("\nTo use these examples:")
    print("  from ripperdoc.examples.config_examples import MULTI_MODEL, create_config_file")
    print("  create_config_file(MULTI_MODEL)")
