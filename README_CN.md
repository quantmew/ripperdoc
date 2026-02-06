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
git clone https://github.com/quantmew/ripperdoc.git
cd ripperdoc

# 从源码安装
pip install -e .
```

## 使用

### 交互模式（推荐）
```bash
ripperdoc
# 或使用短别名
rd
```

这将启动一个交互式会话，您可以：
- 询问关于代码库的问题
- 请求代码修改
- 执行命令
- 导航和探索文件

**选项：**
- `--yolo` - 跳过权限提示（默认启用安全模式）
- `--model <模型名>` - 指定模型（如 `claude-sonnet-4-20250514`、`gpt-4o`）
- `--tools <工具列表>` - 过滤可用工具（逗号分隔，或 "" 表示禁用）
- `--no-mcp` - 禁用 MCP 服务器集成
- `--verbose` - 启用详细日志

### 快速开始

如需引导式介绍，请先阅读本文“交互模式（推荐）”和“Python SDK（无头模式）”章节。

### Python SDK（无头模式）

通过 Python SDK 使用 Ripperdoc，无需终端 UI。查看 [SDK 文档](https://ripperdoc-docs.pages.dev/docs/sdk-overview/) 了解一次性 `query` 助手和基于会话的 `RipperdocClient` 的示例。

#### SDK 示例

- **基础用法**：简单的一次性查询
- **会话管理**：具有上下文的持久化会话
- **工具集成**：直接工具访问和自定义
- **配置**：自定义模型提供商和设置

查看 [SDK 文档](https://ripperdoc-docs.pages.dev/docs/sdk-overview/) 获取完整的 SDK 使用示例。

### 安全模式权限

安全模式默认启用。收到提示时：
- 输入 `y` 或 `yes` 允许单个操作
- 输入 `a` 或 `always` 允许会话期间所有该类操作
- 输入 `n` 或 `no` 拒绝操作

使用 `--yolo` 标志跳过所有权限提示：
```bash
ripperdoc --yolo
```

### Agent Skills

用可复用的 Skill 包扩展 Ripperdoc：

- **个人技能**：`~/.ripperdoc/skills/<技能名>/SKILL.md`
- **项目技能**：`.ripperdoc/skills/<技能名>/SKILL.md`（可提交到 git）
- 每个 `SKILL.md` 以 YAML 头开始：
  - `name` - 技能标识符
  - `description` - 技能功能描述
  - `allowed-tools`（可选）- 限制技能可使用的工具
  - `model`（可选）- 为该技能建议特定模型
  - `max-thinking-tokens`（可选）- 控制思考预算
  - `disable-model-invocation`（可选）- 不调用模型直接使用技能
- 相关文件可放在 `SKILL.md` 同目录
- 技能自动发现并通过 `Skill` 工具按需加载

**内置技能**：PDF 处理（`pdf`）、PowerPoint（`pptx`）、Excel（`xlsx`）

## 示例

### 代码分析
```
> 你能解释这个函数的作用吗？
> 查找所有引用 `parse_config` 函数的地方
```

### 文件操作
```
> 读取 main.py 文件并建议改进
> 创建一个名为 UserProfile.tsx 的新组件
> 更新所有导入以使用新的包结构
```

### 代码生成
```
> 创建一个实现 REST API 客户端的新 Python 脚本
> 为 auth 模块生成单元测试
> 为数据库连接代码添加错误处理
```

### 项目导航
```
> 显示项目中所有的 Python 文件
> 找到用户认证逻辑的实现位置
> 列出项目中所有 API 端点
```

### MCP 集成
```
> 有哪些可用的 MCP 服务器？
> 查询 context7 文档中关于 React hooks 的内容
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
