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
  <a href="https://pypi.org/project/ripperdoc/">
    <img src="https://img.shields.io/badge/version-0.4.4-orange.svg">
  </a>
</p>

</div>

**Ripperdoc** 是一个强大、可扩展的 AI 编程代理,直接在您的终端中运行。受到 [Claude Code](https://claude.com/claude-code)、[Aider](https://github.com/paul-gauthier/aider) 和 [Goose](https://github.com/block/goose) 等工具的启发,Ripperdoc 帮助您通过自然语言对话编写代码、重构项目、执行 shell 命令和管理文件。

## Ripperdoc 的独特之处?

- **🔌 模型无关** - 支持 Anthropic Claude、OpenAI、Google Gemini、DeepSeek 以及任何 OpenAI 兼容 API
- **🎣 可扩展架构** - 26+ 内置工具,配备 hooks 系统用于自定义工作流
- **🤖 多代理协调** - 内置任务图和团队协作,支持复杂工作流
- **📚 技能系统** - 按需加载能力包(PDF、Excel、PowerPoint、自定义语言)
- **🔌 MCP 集成** - 一流支持的模型上下文协议服务器
- **🛡️ 默认安全** - 具有可配置规则和 hooks 的权限系统
- **🎨 美观界面** - 丰富的终端界面,支持主题、语法高亮和交互式 TUI
- **⚡ 后台任务** - 异步运行长时间命令,实时监控

[English](README.md) | [贡献指南](CONTRIBUTING.md) | [文档](https://ripperdoc-docs.pages.dev/)

## 核心功能

### 🛠️ 强大的工具系统
- **26+ 内置工具** - 文件操作(Read、Write、Edit、MultiEdit)、代码搜索(Grep、Glob)、shell 执行(Bash、Background)、LSP 集成等
- **Jupyter 支持** - 直接编辑 .ipynb 笔记本,支持单元格操作
- **后台任务** - 异步运行命令,监控输出和跟踪状态

### 🤖 多代理架构
- **任务图系统** - 持久化任务管理,支持依赖关系、阻塞项和所有权
- **团队协调** - 多代理协作,支持结构化消息传递和协调
- **专业子代理** - 内置代码审查、探索、规划和测试生成的代理

### 🔌 可扩展性
- **技能系统** - 加载 SKILL.md 包来扩展能力(PDF、Excel、PowerPoint、自定义语言)
- **Hooks 系统** - 在生命周期事件执行自定义脚本,支持决策控制
- **自定义命令** - 定义可复用的斜杠命令,支持参数替换
- **MCP 集成** - 连接到模型上下文协议服务器以扩展功能

### 🎨 用户体验
- **丰富的终端 UI** - 美观的界面,支持语法高亮和进度指示器
- **主题支持** - 可定制的配色方案和样式选项
- **交互式 TUI** - 用于管理代理、模型、权限和 hooks 的终端界面
- **安全模式** - 对危险操作具有可配置规则的权限提示

### 💾 会话管理
- **持久化历史** - 完整的对话历史,支持搜索和重放
- **会话分支** - 从任何对话状态创建分支
- **用量跟踪** - 跨会话监控 token 使用和成本

## 安装

### 快速安装
```bash
pip install git+https://github.com/quantmew/ripperdoc.git
```

### 从源码安装
```bash
git clone https://github.com/quantmew/ripperdoc.git
cd ripperdoc
pip install -e .
```

### 开发环境设置
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



## 快速开始

### 启动交互式会话
```bash
ripperdoc
```

### 命令行选项
```bash
ripperdoc [OPTIONS]
```

**选项:**
- `--yolo` - 跳过权限提示(默认启用安全模式)
- `--model <模型名>` - 指定模型(如 `claude-sonnet-4-20250514`、`gpt-4o`)
- `--tools <工具列表>` - 过滤可用工具(逗号分隔,或 "" 表示禁用)
- `--no-mcp` - 禁用 MCP 服务器集成
- `--verbose` - 启用详细日志
- `--theme <主题名>` - 设置 UI 主题

**环境变量:**
- `RIPPERDOC_ENABLE_TASKS=false` - 使用旧版 Todo 工具而非任务图
- `RIPPERDOC_TASK_LIST_ID` - 强制跨会话共享任务图列表 ID
- `RIPPERDOC_MODEL` - 默认使用的模型
- `RIPPERDOC_TEMPERATURE` - 默认温度(0.0-2.0)
- `RIPPERDOC_API_KEY` - 已配置提供商的 API 密钥

任务图作用域行为:
- 默认情况下,任务列表按会话隔离(新会话从干净状态开始;恢复/继续会话保持相同的任务)
- 设置 `RIPPERDOC_TASK_LIST_ID` 以有意跨会话共享一个任务列表

### 基本使用示例

**代码分析:**
```
> 你能解释这个函数的作用吗?
> 查找所有引用 `parse_config` 函数的地方
```

**文件操作:**
```
> 读取 main.py 文件并建议改进
> 创建一个名为 UserProfile.tsx 的新组件
> 更新所有导入以使用新的包结构
```

**代码生成:**
```
> 创建一个实现 REST API 客户端的新 Python 脚本
> 为 auth 模块生成单元测试
> 为数据库连接代码添加错误处理
```

## 高级功能

### 技能系统

使用存储在 `SKILL.md` 文件中的可复用技能包扩展 Ripperdoc:

**技能位置:**
- `~/.ripperdoc/skills/<技能名>/SKILL.md` (个人技能)
- `.ripperdoc/skills/<技能名>/SKILL.md` (项目特定,可提交到 git)

**技能前置数据:**
```yaml
---
name: pdf-processing
description: 全面的 PDF 操作工具包
allowed-tools: Read, Write, Bash
model: claude-sonnet-4-20250514
max-thinking-tokens: 20000
---
```

**内置技能:**
- `pdf` - PDF 操作(提取文本/表格、创建、合并/拆分)
- `pptx` - PowerPoint 演示文稿创建和编辑
- `xlsx` - Excel 电子表格操作,支持公式
- `cangjie` - 仓颉编程语言支持

### Hooks 系统

在生命周期事件执行自定义脚本,支持决策控制:

**Hook 事件:**
- `PreToolUse` - 工具执行前(可阻止/修改)
- `PostToolUse` - 工具执行成功后
- `PostToolUseFailure` - 工具执行失败后
- `PermissionRequest` - 请求权限时
- `UserPromptSubmit` - 用户提交输入时
- `SessionStart/End` - 会话生命周期
- `SubagentStart/Stop` - 子代理生命周期

**Hook 配置:**
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

### MCP 集成

Ripperdoc 支持模型上下文协议以扩展功能:

```bash
# 列出可用的 MCP 服务器
> /mcp

# 查询 MCP 资源
> 有哪些可用的 MCP 服务器?
> 查询 context7 文档中关于 React hooks 的内容
```

### 自定义命令

定义可复用的斜杠命令,支持参数替换:

**命令文件:** `.ripperdoc/commands/deploy.md`
```markdown
---
description: 部署应用到生产环境
---

部署 $ARGUMENTS 到生产环境...

!`npm run build && npm run deploy`
```

**用法:** `/deploy my-feature-branch`

## 斜杠命令

Ripperdoc 提供强大的斜杠命令用于会话管理:

**会话命令:**
- `/exit` - 退出会话
- `/clear` - 清除对话历史
- `/compact` - 压缩对话历史
- `/fork` - 从当前状态创建新会话分支
- `/resume` - 恢复之前的会话

**配置命令:**
- `/config` - 管理配置
- `/models` - 管理模型提供商
- `/tools` - 查看可用工具
- `/permissions` - 管理权限规则
- `/hooks` - 管理 hooks 配置
- `/themes` - 更改 UI 主题
- `/output_language` - 设置输出语言
- `/output_style` - 设置输出样式

**信息命令:**
- `/help` - 显示帮助信息
- `/status` - 显示会话状态
- `/stats` - 显示使用统计
- `/cost` - 显示成本跟踪
- `/doctor` - 运行系统诊断

**功能命令:**
- `/skills` - 列出可用技能
- `/agents` - 管理子代理
- `/tasks` - 任务图管理
- `/todos` - 旧版待办事项管理
- `/commands` - 列出自定义命令
- `/context` - 管理上下文
- `/memory` - 管理记忆
- `/mcp` - MCP 服务器管理

## 项目导航

```
> 显示项目中所有的 Python 文件
> 找到用户认证逻辑的实现位置
> 列出项目中所有 API 端点
> 解释这个代码库的架构
```

## 后台任务

```
> 在后台运行测试
> 启动开发服务器并监控其输出
> 检查后台任务状态
```

## 架构

### 核心组件

- **CLI 层** (`ripperdoc/cli/`) - 终端界面、UI 组件、命令处理器
- **核心层** (`ripperdoc/core/`) - 代理定义、配置、hooks、提供商
- **工具层** (`ripperdoc/tools/`) - 26+ 用于文件操作、代码分析等的内置工具
- **协议层** (`ripperdoc/protocol/`) - 用于 SDK 通信的 Stdio 协议处理器
- **工具层** (`ripperdoc/utils/`) - 用于日志记录、权限、文件操作的共享工具

### 工具类别

**文件操作:**
- `Read` - 读取文件内容,支持可选的偏移量/限制
- `Write` - 创建新文件或覆盖现有文件
- `Edit` - 替换文件中的精确字符串匹配
- `MultiEdit` - 对单个文件进行批量编辑操作
- `NotebookEdit` - 编辑 Jupyter 笔记本单元格

**代码分析:**
- `Grep` - 使用正则表达式模式搜索代码
- `Glob` - 文件模式匹配
- `LSP` - 语言服务器协议集成

**Shell 操作:**
- `Bash` - 执行 shell 命令
- `KillBash` - 终止后台 shell 进程
- `BashOutput` - 从后台任务读取输出

**代理功能:**
- `TaskCreate/Update/Get/List` - 任务图管理
- `TeamCreate/Delete` - 多代理团队协调
- `SendMessage` - 代理间消息传递
- `Task` - 委托给专业子代理

**可扩展性:**
- `Skill` - 按需加载技能包
- `ToolSearch` - 发现和激活工具
- `AskUserQuestion` - 交互式用户提示

### 支持的提供商

- **Anthropic** - Claude 3.5 Sonnet、Claude 3 Opus、Haiku
- **OpenAI** - GPT-4、GPT-4 Turbo、GPT-3.5
- **Google** - Gemini Pro、Gemini Flash
- **DeepSeek** - DeepSeek Coder、DeepSeek Chat
- **自定义** - 任何 OpenAI 兼容 API

## 开发

### 项目结构

```
ripperdoc/
├── cli/              # CLI 界面和 UI 组件
├── core/             # 核心功能和配置
│   ├── hooks/        # Hooks 系统实现
│   ├── providers/    # LLM 提供商实现
│   └── query/        # 查询处理循环
├── tools/            # 内置工具实现
├── protocol/         # Stdio 协议处理器
│   └── stdio/        # 协议实现
├── utils/            # 工具函数
└── data/             # 模型定价和上下文数据
```

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_bash_tool.py

# 运行并生成覆盖率报告
pytest --cov=ripperdoc
```

### 代码质量

```bash
# 类型检查
mypy ripperdoc

# 代码格式化
black ripperdoc

# 代码检查
ruff check ripperdoc

# 使用 Black 格式化
black ripperdoc
```

## 配置

### 配置文件位置
- `~/.ripperdoc/config.json` - 用户级配置
- `.ripperdoc/config.json` - 项目级配置(覆盖用户配置)

### 配置示例

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

## 贡献

我们欢迎贡献!请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

### 开发设置

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 运行测试和代码检查
5. 提交拉取请求

## 更新日志

参见 [CHANGELOG.md](CHANGELOG.md) 了解版本历史和更新。

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。

### 主要许可证条款

- ✅ 商业使用
- ✅ 分发
- ✅ 修改
- ✅ 专利授权
- ✅ 私人使用
- ✅ 子许可
- ❌ 商标使用

## 致谢

灵感来源和构建思路:
- [Claude Code](https://claude.com/claude-code) - Anthropic 官方 CLI
- [aider](https://github.com/paul-gauthier/aider) - AI 结对编程工具
- [Goose](https://github.com/block/goose) - 可扩展 AI 助手
- [Cursor](https://cursor.sh) - AI 驱动的代码编辑器

## 资源

- [文档](https://ripperdoc-docs.pages.dev/)
- [English](README.md)
- [贡献指南](CONTRIBUTING.md)
- [问题跟踪](https://github.com/quantmew/ripperdoc/issues)
- [讨论区](https://github.com/quantmew/ripperdoc/discussions)

---

**由 Ripperdoc 团队用 ❤️ 制作**
