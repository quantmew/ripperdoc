# Quickstart

Get Ripperdoc running in a few minutes.

## 1) Install

From git:
```bash
pip install git+https://github.com/quantmew/ripperdoc.git
```

From source:
```bash
git clone <repository-url>
cd ripperdoc
pip install .
```

## 2) Configure an API key

```bash
export OPENAI_API_KEY="your-api-key-here"
# or for Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 3) Run the CLI

```bash
ripperdoc
```

Optional shortcuts:
```bash
rd
ripperdoc --prompt "Summarize README.md"
```

## 4) Permissions

Safe mode is default. To skip prompts:
```bash
ripperdoc --yolo
```

## 5) Next steps

- Use `/help` to see built-in slash commands.
- See `docs/SDK_USAGE.md` for Python SDK examples.
- Check `docs/` for hooks and configuration details.
