# SGLang parser 误区解释

## 1. 背景

- 现象：`tool_choice=required` 下，响应里只有 `tool_calls`，`content` 为空，同时 `reasoning_content` 也为空。
- 结论：要在 `tool_choice=required` 下拿到 `reasoning_content`，必须开启 `reasoning_parser`。parser 不只是匹配或后处理，它还会改变约束解码在解码期的生效时机。

## 2. 关键术语

| 术语 | 含义 | 关心点 |
| --- | --- | --- |
| `tool_choice=required` | 强制模型必须产出 tool call | 会触发 JSON schema 约束解码 |
| `tool_call_parser` | 把模型输出解析成 OpenAI `tool_calls` | 决定 tool call 解析格式 |
| `reasoning_parser` | 把输出拆成 `reasoning_content` 与 `content` | 同时影响解码期 gating 与返回期拆分 |
| `constrained decoding` | 约束解码，采样时对 logits 做 mask | 约束一旦生效，非 JSON token 会被禁止 |
| `ReasonerGrammarBackend` | 两段式 grammar 包装器 | 在 `think_end_id` 前放行思考段 |

## 3. 调用栈分析

解码期决定模型能不能先输出思考；返回期决定响应字段如何被拆分与清空。

开启 `reasoning_parser` 时，grammar backend 会被包装成两段式 gating：在 `think_end_id` 前不把 JSON schema 约束直接压到输出上，允许模型先输出思考段。

返回期先拆分 `reasoning_content`，再解析 tool calls；required 分支解析成功后会主动清空正文，这是 Serving 层设计行为。

## 总结

- 不开 `reasoning_parser` 时，required 触发的 `json_schema` 约束往往从第一个 token 就生效，模型无法输出思考与正文，最后 required 后处理还会清空正文，所以只剩 `tool_calls`。
- 开了 `reasoning_parser` 时，会启用两段式 gating，允许先输出思考段，再进入 JSON 约束，返回期再把思考拆到 `reasoning_content`。

