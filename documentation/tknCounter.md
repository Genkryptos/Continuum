# TknCounter overview

`TknCounter` provides a provider-aware way to estimate the number of tokens a list of chat-style messages will consume.
It chooses a counting strategy based on the `model` name and whether direct provider counting is enabled.

## How it works

1. **OpenAI GPT-family models (`gpt-…`)**
   * Uses OpenAI-style chat counting rules (per-message and per-name token overhead) and `tiktoken` encodings.
   * Falls back to a default encoding (`cl100k_base`) if the model-specific tokenizer is unknown.

2. **Anthropic Claude models (`claude-…`)**
   * When `use_provider_api=True`, calls `beta.messages.count_tokens` to mirror Anthropic's server-side accounting.
   * If the Anthropic client cannot be initialized while provider counting is requested, a clear error is raised when counting is attempted.
   * With provider counting disabled, falls back to `litellm.token_counter`.

3. **Other model families (Gemini, Cohere, Mistral, etc.)**
   * Explicitly routed through `litellm.token_counter` so future families can be added without changing call sites.

## Public helpers

* `count_request_tokens(model, messages)` — counts input/request tokens (alias of `count_tokens`).
* `estimate_total_tokens(model, messages, max_output_tokens=None)` — combines input tokens with an optional output budget to project total usage.

## Usage

```python
from helper.tknCounter import TknCounter

counter = TknCounter(use_provider_api=True)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this conversation."},
]

input_tokens = counter.count_request_tokens(model="gpt-4o", messages=messages)
estimated_total = counter.estimate_total_tokens(
    model="gpt-4o",
    messages=messages,
    max_output_tokens=200,
)
```

Toggle `use_provider_api` to enable or disable Anthropic’s server-side token counting for Claude models.
When disabled, Claude requests also use the `litellm` fallback.