<div align="center">

# Ripperdoc

_开源、可扩展的 AI 编程代理，在终端中运行_

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
</p>
</div>

Ripperdoc 是你的本地 AI 编程助手，类似于 [Claude Code](https://claude.com/claude-code)、[Codex](https://github.com/openai/codex)、[Gemini CLI](https://github.com/google-gemini/gemini-cli)、[Aider](https://github.com/paul-gauthier/aider) 和 [Goose](https://github.com/block/goose)。它可以编写代码、重构项目、执行 shell 命令、管理文件——全部通过终端中的自然语言对话完成。

设计上追求最大灵活性，Ripperdoc 支持**任意 LLM**（Anthropic Claude、OpenAI、DeepSeek、通过 OpenAI 兼容 API 的本地模型），支持**自定义 hooks** 来拦截和控制工具执行，并提供交互式 CLI 和 **Python SDK** 用于无头自动化。

## 功能特性

- **AI 智能助手** - 使用 AI 模型理解和响应编码请求
- **多模型支持** - 支持 Anthropic Claude 和 OpenAI 模型
- **代码编辑** - 通过智能建议直接编辑文件
- **代码库理解** - 分析项目结构和代码关系
- **命令执行** - 运行 shell 命令并获得实时反馈
- **工具系统** - 可扩展架构，支持专用工具
- **Agent Skills** - 通过 SKILL.md 按需扩展能力
- **子代理系统** - 将任务委托给具有自己工具范围的专用代理
- **丰富 UI** - 美观的终端界面，支持语法高亮
- **文件操作** - 读取、写入、编辑、搜索和管理文件
- **任务跟踪** - 为每个项目规划、读取和更新持久化任务列表
- **后台命令** - 在后台运行命令并监控输出
- **权限系统** - 安全模式，操作需要权限确认
- **批量编辑支持** - 对文件进行批量编辑操作
- **MCP 服务器支持** - 与 Model Context Protocol 服务器集成
- **会话管理** - 持久化会话历史和用量跟踪
- **Jupyter Notebook 支持** - 直接编辑 .ipynb 文件
- **Hooks 系统** - 在生命周期事件执行自定义脚本，支持决策控制
- **自定义命令** - 定义可复用的斜杠命令，支持参数替换

## 安装

### 快速安装
从 git 仓库安装：
```bash
pip install git+https://github.com/quantmew/ripperdoc.git
```

或从源码安装：
```bash
# 克隆仓库
git clone <repository-url>
cd ripperdoc

# 从源码安装
pip install -e .
```

### 配置

设置 API 密钥作为环境变量：
```bash
export OPENAI_API_KEY="your-api-key-here"
# 或用于 Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 使用

### 交互模式（推荐）
```bash
ripperdoc
```

这将启动一个交互式会话，您可以：
- 询问关于代码库的问题
- 请求代码修改
- 执行命令
- 导航和探索文件

### 快速开始

如需引导式介绍，请查看 [QUICKSTART.md](QUICKSTART.md) 指南。

### Python SDK（无头模式）

通过新的 Python SDK 使用 Ripperdoc，无需终端 UI。查看 [docs/SDK_USAGE.md](docs/SDK_USAGE.md) 了解一次性 `query` 助手和基于会话的 `RipperdocClient` 的示例。中文指南见 [docs/SDK_USAGE_CN.md](docs/SDK_USAGE_CN.md)。

#### SDK 示例

- **基础用法**：简单的一次性查询
- **会话管理**：具有上下文的持久化会话
- **工具集成**：直接工具访问和自定义
- **配置**：自定义模型提供商和设置

查看 [examples/](examples/) 目录获取完整的 SDK 使用示例。

### 安全模式权限

安全模式是默认设置。使用 `--yolo` 跳过权限提示。选择 `a`/`always` 允许在当前会话中使用某个工具（不会跨会话持久化）。

### Agent Skills

用可复用的 Skill 包扩展 Ripperdoc：

- 个人技能放在 `~/.ripperdoc/skills/<技能名>/SKILL.md`
- 项目技能放在 `.ripperdoc/skills/<技能名>/SKILL.md`，可随仓库一起提交
- 每个 `SKILL.md` 以 YAML 头开始：`name`、`description`，可选 `allowed-tools`、`model`、`max-thinking-tokens`、`disable-model-invocation`，正文写具体指令；相关文件可放在同目录
- 通过 `Skill` 工具加载后，技能中的模型和思考预算提示会自动作用于当前会话
- Ripperdoc 会在系统提示中列出技能名称和描述，匹配时通过 `Skill` 工具自动加载完整内容

## 示例

### 代码分析
```
> 你能解释这个函数的作用吗？
```

### 文件操作
```
> 读取 main.py 文件并建议改进
```

### 代码生成
```
> 创建一个实现 REST API 客户端的新 Python 脚本
```

### 项目导航
```
> 显示项目中所有的 Python 文件
```

## 开发

### 设置开发环境
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 类型检查
mypy ripperdoc

# 代码格式化
black ripperdoc

# 代码检查
ruff ripperdoc
```



## 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 主要许可证条款

- **商业使用**：允许
- **分发**：允许
- **修改**：允许
- **专利授权**：包含
- **私人使用**：允许
- **子许可**：允许
- **商标使用**：不授予

有关完整的许可证条款和条件，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

灵感来源：
- [Claude Code](https://claude.com/claude-code) - Anthropic 官方 CLI
- [aider](https://github.com/paul-gauthier/aider) - AI 结对编程工具
