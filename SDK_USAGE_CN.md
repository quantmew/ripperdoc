# Ripperdoc Python SDK 使用指南（无界面调用）

本指南介绍如何在纯 Python 环境中调用 Ripperdoc，而无需启动终端 UI。提供一次性调用的 `query` 辅助函数和可持续对话的 `RipperdocClient`。

## 适用场景
- 在 CI / 后台任务中直接触发代码分析或修改
- 在自建 REPL / Web 服务里复用同一对话上下文
- 需要程序化控制工具白名单、工作目录或权限策略

## 前置条件
1) 安装项目依赖：`pip install -e .`  
2) 配置模型凭据（环境变量或 `~/.ripperdoc.json`，与 CLI 相同）：
```bash
export ANTHROPIC_API_KEY=your-key
# 或
export OPENAI_API_KEY=your-key
```

## 快速开始：一次性调用
```python
import asyncio
from ripperdoc.sdk import query, RipperdocOptions
from ripperdoc.utils.messages import AssistantMessage, ProgressMessage


async def main():
    options = RipperdocOptions(
        safe_mode=False,                # 关闭交互式权限提示，方便无人值守
        allowed_tools=["Bash", "View"], # 限制可用工具
        cwd="/path/to/project",         # 固定工作目录
    )

    async for msg in query("列出项目里的 Python 文件", options=options):
        if isinstance(msg, AssistantMessage):
            print("assistant:", msg.message.content)
        elif isinstance(msg, ProgressMessage):
            print("progress:", msg.content)


asyncio.run(main())
```

## 会话模式：复用上下文
```python
import asyncio
from ripperdoc.sdk import RipperdocClient, RipperdocOptions
from ripperdoc.utils.messages import AssistantMessage


async def main():
    async with RipperdocClient(RipperdocOptions(safe_mode=False)) as client:
        await client.query("总结 README.md 的作用")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                print(msg.message.content)

        await client.query("再告诉我 CLI 入口文件在哪里")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                print(msg.message.content)


asyncio.run(main())
```

## 配置项速查（`RipperdocOptions`）
- `tools`: 自定义工具列表；默认使用内置工具集。
- `allowed_tools` / `disallowed_tools`: 名称白/黑名单，过滤后自动重建 Task 工具，确保子代理只看到允许的工具。
- `safe_mode`: True 时复用 CLI 的权限询问逻辑；False 适合无人值守。
- `permission_checker`: 自定义检查函数，签名为 `(tool, parsed_input) -> bool | PermissionResult`，可异步。
- `cwd`: 会话生效的工作目录，断开连接后自动恢复。
- `system_prompt`: 完全替换系统提示；`additional_instructions`: 在默认提示后追加文字。
- `context`: 额外上下文字典，会拼接进系统提示。
- `model`, `max_thinking_tokens`, `verbose`: 透传到核心查询上下文。

## 工具过滤与 Task 子代理
使用 `allowed_tools`/`disallowed_tools` 过滤后，SDK 会重建默认的 Task 工具，使子代理仅能看到过滤后的基础工具集合，避免越权。

## 权限控制
默认 `safe_mode=False`，不会弹出交互式确认。若需要自定义策略，可提供 `permission_checker`：
```python
async def allow_only_reads(tool, parsed_input):
    if tool.is_read_only():
        return True
    return False  # 拒绝写操作

options = RipperdocOptions(permission_checker=allow_only_reads, safe_mode=True)
```

## 提示词与记忆
- 不传 `system_prompt` 时，SDK 会调用 `build_system_prompt` 并自动注入 `RIPPERDOC.md`/`RIPPERDOC.local.md` 等记忆文件。
- 追加指令：`additional_instructions="你是该仓库的守护者，回答请用要点。"`

## 返回的消息类型
`query` / `RipperdocClient.receive_*` 产出三类消息（来自 `ripperdoc.utils.messages`）：  
- `UserMessage`：用户输入或工具结果  
- `AssistantMessage`：模型回复，`message.content` 可为字符串或内容块列表  
- `ProgressMessage`：工具执行进度

## 调试与常见问题
- 需要查看执行细节时设置 `verbose=True`。  
- 若在 CI 中需阻止网络或写操作，请用 `allowed_tools` 或自定义 `permission_checker` 限制。  
- 终止长任务：`await client.interrupt()`。
