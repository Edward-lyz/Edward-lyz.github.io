# 【工程】SGLang parser 误区解释

# 1. 背景

- 现象：`tool_choice=required` 下，响应里只有 `tool_calls`，`content` 为空，同时 `reasoning_content` 也为空。
- 结论：要在 `tool_choice=required`下拿到 `reasoning_content`，必须开启 `reasoning_parser`，否则必然拿不到。parser 不仅是匹配或后处理的功能，它还会改变约束解码在解码期的生效时机。required 之所以让 `content` 为空，是 Serving 层在解析出 tool calls 后主动清空正文，这是设计行为。

# 2. 术语表

| 术语 | 含义 | 你关心的点 |
| --- | --- | --- |
| `tool_choice=required` | 强制模型必须产出 tool call | 会触发 JSON schema 约束解码 |
| `tool_call_parser` | 把模型输出解析成 OpenAI `tool_calls` | 决定 tool call 的解析格式 |
| `reasoning_parser` | 把输出拆成 `reasoning_content` 与 `content` | 同时影响解码期 gating 与返回期拆分 |
| `constrained decoding` | 约束解码，采样时对 logits 做 mask | 约束一旦生效，非 JSON token 会被禁止 |
| `json_schema` | 用 JSON schema 强约束输出结构 | required 默认走它 |
| `ReasonerGrammarBackend` | 两段式 grammar 包装器 | 在 `think_end_id` 前放行思考段 |
| `think_end_id` | 结束思考段的 token id | 决定何时开始强制 JSON |
| `require_reasoning` | 请求级布尔值，表示需要思考段 | 决定 gating 初始状态 |

# 3. 调用栈分析

这里把顺序拆成两段时期进行讨论：

- 解码期，决定模型能不能先输出思考 — 实际代码包含该功能
- 返回期，决定响应字段如何被拆分与清空 — 之前对 parser 的认识有错误，认为只会做拆分

## 解码期调用栈

1. HTTP 入口把请求交给 Serving
2. ServingChat 发现 required，选择 json_schema 约束

```python
# required 或指定函数会使用 json_schema
if tool_choice is required or is specific function:
	tool_call_constraint = ('json_schema', schema)
```

1. ServingChat 同时计算是否需要思考段

```python
# require_reasoning 是布尔值
require_reasoning = decide_from_request(enable_thinking or thinking)
```

1. `TokenizerManager` 把 `tokenized request` 发给 **Scheduler**
2. `SchedulerOutputProcessorMixin` 会对 `next_token_id` 进行处理：

```python
if req.grammar is not None:
  # FIXME: this try-except block is for handling unexpected xgrammar issue.
  try:
      if batch.spec_algorithm.is_none():
          # Normal decode: single token
          **req.grammar.accept_token(next_token_id)**
      elif batch.is_spec_v2:
          # Speculative decode: next_token_id is a list of accepted tokens
          for token_id in next_token_id:
              **req.grammar.accept_token(token_id)**
  except ValueError as e:
      # Grammar accept_token can raise ValueError if the token is not in the grammar.
      # This can happen if the grammar is not set correctly or the token is invalid.
      logger.error(
          f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
      )
      self.abort_request(AbortReq(rid=req.rid))
```

<aside>
⚠️

1. 开启 `reasoning_parser` 时，**grammar backend 会被包装成两段式 gating**
</aside>

- 位置：`python/sglang/srt/constrained/base_grammar_backend.py`
- 条件：server 启动参数设置了 `--reasoning-parser`，并且 tokenizer 具备 `think_end_id`

```python
class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(self, grammar: BaseGrammarObject, think_end_id: int):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        # -1    means thinking has not ended yet
        # 0     means just ended thinking in the last token
        # +     means number of tokens after thinking ended
        self.tokens_after_think_end = -1

    def maybe_init_reasoning(self, reasoning: bool):
        self.tokens_after_think_end = -1 if reasoning else 0

    def transfer_state(self, token: int) -> int:
        if self.tokens_after_think_end == -1 and token == self.think_end_id:
            self.tokens_after_think_end = 0
        elif self.tokens_after_think_end >= 0:
            self.tokens_after_think_end += 1

    def rollback_state(self):
        if self.tokens_after_think_end == 0:
            self.tokens_after_think_end = -1
        elif self.tokens_after_think_end > 0:
            self.tokens_after_think_end -= 1

    **def accept_token(self, token: int):
        if self.tokens_after_think_end >= 0:
            self.grammar.accept_token(token)
        self.transfer_state(token)**

    def rollback(self, k):
        steps_after_think = min(k, self.tokens_after_think_end)
        if steps_after_think > 0:
            self.grammar.rollback(steps_after_think)

        for _ in range(k):
            self.rollback_state()

class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key, reasoning)
        # avoid wrapping invalid grammar, so that the scheduler can detect it
        if ret is None or ret is INVALID_GRAMMAR_OBJ:
            return ret
        obj = ReasonerGrammarObject(ret, self.think_end_id)
        obj.maybe_init_reasoning(reasoning)
        return obj
```

可以看到，在初始化 `ReasonerGrammarBackend` 时，会调用一个 `maybe_init_reasoning` 函数，该函数会尝试性地让模型先思考，且每一步 `accept_token` 时，都会根据是否结束思考进行判断。

## 返回期调用栈

返回期只做两件事。

- 拆分 reasoning_content
- 解析 tool_calls 并清空正文
1. ServingChat 拿到完整文本后先做 reasoning 拆分
2. 然后解析 tool calls
3. required 分支解析成功后会返回空正文
- 逻辑位置：`python/sglang/srt/entrypoints/openai/serving_chat.py`

```python
if tool_choice is required:
	tool_calls = json.loads(text)
	return tool_calls, '', finish_reason
```

1. 最终字段落地
- `reasoning_content` 取决于步骤 1 是否执行成功
- `content` 在 required 成功解析后会被置空
- `tool_calls` 在 required 或 tool_call_parser 解析成功后会填充

# 总结

- 不开 `reasoning_parser` 时，required 触发的 `json_schema` 约束往往从第一个 token 就生效，模型无法输出思考与正文，最后 required 后处理还会清空正文，所以只剩 `tool_calls`。
- 开了 `reasoning_parser` 时，会启用两段式 gating，允许先输出思考段，再进入 JSON 约束，返回期再把思考拆到 `reasoning_content`。