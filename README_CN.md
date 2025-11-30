# Ripperdoc - AI 驱动的终端助手

Ripperdoc 是一个 AI 驱动的终端助手，专为编码任务设计，提供交互式界面用于 AI 辅助开发、文件管理和命令执行。

## 功能特性

- **AI 智能助手** - 使用 AI 模型理解和响应编码请求
- **多模型支持** - 支持 Anthropic Claude 和 OpenAI 模型
- **代码编辑** - 通过智能建议直接编辑文件
- **代码库理解** - 分析项目结构和代码关系
- **命令执行** - 运行 shell 命令并获得实时反馈
- **工具系统** - 可扩展架构，支持专用工具
- **子代理系统** - 将任务委托给具有自己工具范围的专用代理
- **丰富 UI** - 美观的终端界面，支持语法高亮
- **文件操作** - 读取、写入、编辑、搜索和管理文件
- **任务跟踪** - 为每个项目规划、读取和更新持久化任务列表
- **后台命令** - 在后台运行命令并监控输出
- **权限系统** - 安全模式，操作需要权限确认
- **批量编辑支持** - 对文件进行批量编辑操作
- **MCP 服务器支持** - 与 MCP（Model Context Protocol）服务器集成，扩展工具能力

## 可用工具

- **Bash** - 执行 shell 命令
- **BashOutput** - 读取后台命令的输出
- **KillBash** - 终止后台命令
- **View** - 读取文件内容
- **Edit** - 通过精确匹配编辑文件
- **MultiEdit** - 对文件进行批量编辑操作
- **NotebookEdit** - 编辑 Jupyter notebook 文件
- **Write** - 创建新文件
- **Glob** - 查找匹配模式的文件
- **Grep** - 在文件中搜索模式
- **LS** - 列出目录内容
- **TodoRead** - 读取当前任务列表或下一个可执行任务
- **TodoWrite** - 创建和更新持久化任务列表
- **Task** - 将任务委托给专用子代理
- **ListMcpServers** - 列出配置的 MCP 服务器及其工具
- **ListMcpResources** - 列出 MCP 服务器提供的资源
- **ReadMcpResource** - 从 MCP 服务器读取特定资源

## 项目结构

```
ripperdoc/
├── core/                    # 核心功能
│   ├── tool.py             # 基础工具接口
│   ├── query.py            # AI 查询系统
│   ├── config.py           # 配置管理
│   ├── commands.py         # 命令定义
│   ├── permissions.py      # 权限系统
│   ├── agents.py           # 代理管理
│   ├── default_tools.py    # 默认工具配置
│   └── system_prompt.py    # 系统提示
├── tools/                  # 工具实现
│   ├── bash_tool.py
│   ├── bash_output_tool.py
│   ├── kill_bash_tool.py
│   ├── file_edit_tool.py
│   ├── multi_edit_tool.py
│   ├── notebook_edit_tool.py
│   ├── file_read_tool.py
│   ├── file_write_tool.py
│   ├── glob_tool.py
│   ├── grep_tool.py
│   ├── ls_tool.py
│   ├── todo_tool.py
│   ├── task_tool.py
│   ├── mcp_tools.py
│   └── background_shell.py
├── utils/                  # 工具函数
│   ├── messages.py
│   ├── message_compaction.py
│   ├── log.py
│   ├── todo.py
│   ├── memory.py
│   ├── mcp.py
│   ├── session_history.py
│   └── session_usage.py
├── cli/                    # CLI 接口
│   ├── cli.py             # 主 CLI 入口点
│   ├── commands/          # 命令实现
│   │   ├── agents_cmd.py
│   │   ├── base.py
│   │   ├── clear_cmd.py
│   │   ├── compact_cmd.py
│   │   ├── config_cmd.py
│   │   ├── context_cmd.py
│   │   ├── cost_cmd.py
│   │   ├── exit_cmd.py
│   │   ├── help_cmd.py
│   │   ├── mcp_cmd.py
│   │   ├── models_cmd.py
│   │   ├── resume_cmd.py
│   │   ├── status_cmd.py
│   │   └── tools_cmd.py
│   └── ui/                # UI 组件
│       ├── rich_ui.py     # Rich 终端 UI
│       ├── context_display.py
│       ├── spinner.py     # 加载动画
│       └── helpers.py
├── sdk/                   # Python SDK
│   ├── client.py
│   └── __init__.py
└── examples/              # 示例代码
    ├── config_examples.py
    └── tool_examples.py
```

## 安装

### 快速安装
```bash
# 克隆仓库
git clone <repository-url>
cd Ripperdoc

# 从源码安装
pip install -e .

# 或使用安装脚本
./install.sh
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

通过新的 Python SDK 使用 Ripperdoc，无需终端 UI。查看 [SDK_USAGE.md](SDK_USAGE.md) 了解一次性 `query` 助手和基于会话的 `RipperdocClient` 的示例。中文指南见 [SDK_USAGE_CN.md](SDK_USAGE_CN.md)。

### MCP 服务器支持

Ripperdoc 支持 MCP（Model Context Protocol）服务器集成，可以动态加载和使用 MCP 服务器提供的工具和资源。使用 `ListMcpServers`、`ListMcpResources` 和 `ReadMcpResource` 工具来探索和使用 MCP 功能。

### 安全模式权限

安全模式是默认设置。使用 `--unsafe` 跳过权限提示。选择 `a`/`always` 允许在当前会话中使用某个工具（不会跨会话持久化）。

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

### 项目文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [DEVELOPMENT.md](DEVELOPMENT.md) - 开发指南
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [PYTERMGUI_USAGE.md](PYTERMGUI_USAGE.md) - PyTermGUI 使用示例
- [CHANGELOG.md](CHANGELOG.md) - 发布历史
- [TODO.md](TODO.md) - 当前开发任务

## 许可证

Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

灵感来源：
- [Claude Code](https://claude.com/claude-code) - Anthropic 官方 CLI
- [aider](https://github.com/paul-gauthier/aider) - AI 结对编程工具