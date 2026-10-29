# Model access and credentials

[Home](../README.md) · [Installation](installation.md) · [Evaluation](evaluation.md)

The model client is [remote_model.py](../responsible_robot_bench/planner/remote_model.py).
Choose a backend with `--model_name` and `--model_type remote`. The client retains
provider-specific message formatting and response parsing.

| Backend | Model-name branch | API-key environment variable |
| --- | --- | --- |
| OpenAI | Name containing `gpt` | `OPENAI_API_KEY` |
| Anthropic | Name containing `claude` | `ANTHROPIC_API_KEY` |
| Gemini | Name containing `gemini` | `GEMINI_API_KEY` |
| DashScope | Lowercase name containing `qwen` | `DASHSCOPE_API_KEY` |
| Local Qwen2-VL | Supported `Qwen2-VL-7B-Instruct` or `Qwen2-VL-72B-Instruct` name | `RRB_API_KEY`; optional for a local endpoint |
| Fireworks Llama | Name containing `90b-vision-instruct` | `firework_API_KEY` |

Use an available model name from your provider. The current routing is
case-sensitive; a generic compatible endpoint does not imply support for every
model's response format. Llama and InternVL branches and the `lmdeploy` local
path also remain in the client.

For example, configure GPT in your shell, then use `--model_name "$RRB_MODEL"`
with any of the three evaluation commands:

```bash
export OPENAI_API_KEY="<your API key>"
export RRB_MODEL="gpt-4o"
```

For Claude, Gemini, or DashScope, use the matching key above and model name.
Claude additionally requires its SDK: `.venv/bin/python -m pip install anthropic`.
Do not put actual keys in tracked files.

For a local Qwen2-VL server:

```bash
export RRB_BASE_URL="http://127.0.0.1:18017/v1"
export RRB_MODEL="Qwen/Qwen2-VL-7B-Instruct"
```

`RRB_MODEL` is a shell variable passed through `--model_name`. The client reads
`RRB_BASE_URL` when Python starts; `remote_url` takes precedence if also set.
Provide the base URL without `/chat/completions`. The local Qwen2-VL client uses
`RRB_API_KEY` or the placeholder `EMPTY` when that variable is unset.

If you maintain a local `.env` file, create it yourself and load it before
starting Python; the evaluator does not load it automatically. Do not commit
credentials or put them in prompts, examples, endpoint URLs, or shared logs.
Review model responses, observations, and execution records before sharing.

The optional `tools/serve_qwen.py` service has no authentication. Keep it bound
to loopback and use an SSH tunnel for a remote simulator. It supports CONFIG-11
with five example images plus the current image and checks the complete input
and requested output against the loaded model's context capacity. See the
[installation guide](installation.md#5-configure-model-inference) for deployment.
