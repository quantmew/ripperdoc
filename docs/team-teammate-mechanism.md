# Team / Teammate 机制说明

本文档描述 Ripperdoc 中 `TeamCreate`、`Task(team_name+teammate_name)`、`SendMessage`、`TeamDelete` 的协作机制与协议行为。

## 1. 术语与目标

- `Team`: 一个协作域，绑定成员名单、消息通道、共享任务列表。
- `Teammate`: 团队内具名成员（如 `agent-a`）。
- `Team Lead`: 默认协调者（通常为 `team-lead`）。
- `SendMessage`: 团队内协议消息入口（点对点、广播、shutdown、plan approval）。

核心目标：

- 保证消息有明确收件人和可追踪协议语义。
- 保证子代理可见到带来源信息的 teammate 消息。
- 保证 shutdown 流程是“请求 -> 响应 -> 退出”的协议闭环。

## 2. 生命周期概览

1. `TeamCreate` 创建团队配置与任务目录。
2. 使用 `Task` 通过 `team_name + teammate_name` 启动/恢复成员。
3. 使用 `SendMessage` 在成员间传递协议消息。
4. Query 循环自动注入 inbox 未读消息给对应参与者。
5. 所有成员下线后执行 `TeamDelete` 清理团队与任务资源。

## 3. Team 与成员状态

### 3.1 TeamCreate

- 创建 `~/.ripperdoc/teams/{team}/config.json`
- 创建 `~/.ripperdoc/tasks/{team}/`
- 会设置当前 active team context

### 3.2 成员身份绑定（强烈建议）

请始终通过以下方式启动成员：

- `Task(..., team_name="team-smoke", teammate_name="agent-a")`
- `Task(..., team_name="team-smoke", teammate_name="agent-b")`

这样可以保证：

- 成员名与 inbox 路由一致
- 消息注入目标不漂移
- shutdown/TeamDelete 的 active 状态可正确追踪

## 4. SendMessage 协议语义

支持类型：

- `message`
- `broadcast`
- `shutdown_request`
- `shutdown_response`
- `plan_approval_response`

### 4.1 recipient 规范化与校验

- 支持 `name@team` 别名（如 `AGENT-B@team-smoke`）。
- 支持大小写不敏感匹配到团队 canonical 名称。
- 若 recipient 不在 team roster（且不是 `team-lead`），调用直接失败（`Unknown recipient ...`），不会“静默成功”。

### 4.2 summary 规则

- `message` / `broadcast` 仍要求 `summary` 非空。
- 不再限制 summary 必须 5-10 词。

### 4.3 shutdown 协议

- `shutdown_request` 内容为结构化 JSON（含 `type/request_id/sender/content`）。
- teammate 需要显式发送 `shutdown_response`（`approve=true/false`）。
- 系统根据 `shutdown_response`（批准）进入退出路径。

## 5. 消息存储与注入

## 5.1 存储模型

- 每个团队有消息日志（jsonl）与成员 inbox（json）。
- 未命中在线 listener 时写入 inbox；命中 listener 时直接入队 pending queue。

## 5.2 自动注入行为

Query 循环会在迭代时：

1. 读取当前参与者 inbox 未读消息。
2. 优先注入 `shutdown_request`（高优先级协议消息）。
3. 将消息包装为可见格式：

```xml
<teammate-message teammate_id="sender" summary="...">
message body
</teammate-message>
```

4. 同步附带 metadata（team/sender/recipient/request_id 等）供工具逻辑消费。

这保证模型既能“看到文本来源”，也能“读到结构化元数据”。

## 6. 子代理退出（shutdown）机制

推荐流程：

1. lead 发送 `shutdown_request` 给目标 teammate。
2. teammate 在其任务上下文中调用 `SendMessage(type=shutdown_response, request_id=..., approve=true)`。
3. 子代理检测到批准的 `shutdown_response` 对应 tool_result 后退出。

注意：

- 不应把“收到 shutdown_request 就立即强制取消”当作标准协议路径。
- 正确路径是“显式响应后退出”，以保证可审计性。

## 7. TeamDelete 约束

`TeamDelete` 会在有 active 成员时拒绝删除。

- 成功前提：成员已退出或标记 inactive。
- 失败返回应明确阻塞成员，便于逐个 shutdown。

## 8. 故障分类口径（测试/排障）

- `未入队`: SendMessage 调用失败，或 recipient 非法。
- `未注入`: 消息已入队，但目标成员新轮次未看到 teammate-message。
- `未消费`: 已看到消息，但未按要求执行（如不回执、不处理）。
- `未下线`: shutdown 流程走完后成员仍持续运行。
- `TeamDelete 被阻塞`: 删除失败且存在 active 成员。

## 9. 推荐冒烟测试策略

1. 固定 `team_name + teammate_name` 启动成员。
2. 先测 `message`（点对点），再测 `broadcast`。
3. 验证注入时必须检查是否出现 `<teammate-message ...>`。
4. shutdown 必须校验：
- request_id 是否匹配
- 是否出现 `shutdown_response`
- 成员是否真正退出
5. 最后执行 TeamDelete 验证清理与阻塞逻辑。

## 10. 相关实现位置

- `ripperdoc/tools/team_tool.py`
- `ripperdoc/tools/task_tool.py`
- `ripperdoc/core/query/loop.py`
- `ripperdoc/utils/teams.py`
- `ripperdoc/core/hooks/state.py`

