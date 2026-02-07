# Migration Guide: v0.3.3 → v0.4.0

**⚠️ IMPORTANT: This is a major release with breaking changes. Please read this guide carefully before upgrading.**

## Overview

Version 0.4.0 introduces significant architectural improvements including a new model catalog system, task graph features, and team collaboration capabilities. These changes require updates to your configuration files.

## Breaking Changes

### 1. Configuration File Format Changes

The `provider` field has been renamed to `protocol` throughout the codebase.

**Old format (v0.3.3):**
```json
{
  "models": [
    {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "api_key": "sk-ant-...",
      "max_tokens": 4096
    }
  ]
}
```

**New format (v0.4.0):**
```json
{
  "models": [
    {
      "protocol": "anthropic",
      "model": "claude-sonnet-4-5-20250929",
      "api_key": "sk-ant-...",
      "max_tokens": 4096
    }
  ]
}
```

### 2. Provider Type Renaming

The following provider types have been consolidated into protocol types:

| Old Provider (v0.3.3) | New Protocol (v0.4.0) |
|----------------------|----------------------|
| `openai` | `openai_compatible` |
| `openai-compatible` | `openai_compatible` |
| `mistral` | `openai_compatible` |
| `deepseek` | `openai_compatible` |
| `kimi` | `openai_compatible` |
| `qwen` | `openai_compatible` |
| `glm` | `openai_compatible` |
| `google` | `gemini` |

### 3. Model Profile Structure

Model profiles now support additional optional fields that auto-populate from the built-in model catalog:

```json
{
  "protocol": "anthropic",
  "model": "claude-sonnet-4-5-20250929",
  "api_key": "sk-ant-...",
  "max_tokens": 4096,
  "temperature": 1.0,
  "context_window": 200000,
  "max_input_tokens": 200000,
  "max_output_tokens": 8192,
  "mode": "chat",
  "supports_reasoning": false,
  "supports_vision": true,
  "price": {
    "input": 3.0,
    "output": 15.0
  },
  "currency": "USD"
}
```

**Note:** All new fields are optional. If omitted, they will be auto-populated from the model catalog when available.

## Migration Steps

### Step 1: Backup Your Configuration

Before upgrading, backup your configuration files:

```bash
# Backup user configuration
cp ~/.ripperdoc/config.json ~/.ripperdoc/config.json.backup

# Backup any project-specific configurations
find . -name "config.json" -path "*/.ripperdoc/*" -exec cp {} {}.backup \;
```

### Step 2: Upgrade Ripperdoc

```bash
pip install --upgrade ripperdoc
```

Or if using uv:

```bash
uv pip install --upgrade ripperdoc
```

### Step 3: Update Configuration Files

Update your `config.json` files:

1. **Replace `provider` with `protocol`**
2. **Update provider type names** using the table above
3. **Optionally add new model profile fields** (recommended for better cost tracking)

Example migration script:

```bash
# Backup and update config
python3 << 'EOF'
import json
import os
from pathlib import Path

# Provider to protocol mapping
provider_to_protocol = {
    "openai": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "mistral": "openai_compatible",
    "deepseek": "openai_compatible",
    "kimi": "openai_compatible",
    "qwen": "openai_compatible",
    "glm": "openai_compatible",
    "google": "gemini",
}

config_path = Path.home() / ".ripperdoc" / "config.json"
backup_path = config_path.with_suffix('.json.backup')

# Read configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Update models
if 'models' in config:
    for model in config['models']:
        if 'provider' in model:
            old_provider = model['provider']
            # Map to new protocol name
            new_protocol = provider_to_protocol.get(old_provider, old_provider)
            model['protocol'] = new_protocol
            del model['provider']
            print(f"Updated: {old_provider} -> {new_protocol}")

# Write updated configuration
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration updated successfully!")
EOF
```

### Step 4: Verify Configuration

Start Ripperdoc and verify your configuration:

```bash
ripperdoc
```

Check that your models are correctly configured:

```bash
ripperdoc /models
```

## New Features to Explore

After migration, explore these new features:

1. **Task Graph**: Use `TaskCreate`, `TaskList`, `TaskUpdate` for task management
2. **Team Collaboration**: Use `TeamCreate`, `SendMessage` for multi-agent coordination
3. **Output Styles**: Try `/output-style` to customize response formatting
4. **Output Language**: Use `/output-language` to control AI response language
5. **Custom Commands**: Use `/commands` to manage file-based custom commands
6. **Working Directories**: Use `/add-dir` to add multiple working directories

## Rollback Instructions

If you encounter issues and need to rollback:

```bash
# Restore previous version
pip install ripperdoc==0.3.3

# Restore configuration
mv ~/.ripperdoc/config.json.backup ~/.ripperdoc/config.json
```

## Getting Help

If you need assistance with migration:

1. Check the [GitHub Issues](https://github.com/quantmew/Ripperdoc/issues) for known migration issues
2. Review the [main documentation](README.md) for configuration details
3. Run `/doctor` command to diagnose configuration issues

## Summary of Changes

- ✅ New model catalog with 200+ models
- ✅ Task graph and team collaboration system
- ✅ Output style and language customization
- ✅ Enhanced hooks with async support
- ✅ Improved session management with sidecar index
- ✅ Better type annotations and code organization
- ⚠️ **Breaking:** `provider` → `protocol` rename
- ⚠️ **Breaking:** Configuration file format changes
