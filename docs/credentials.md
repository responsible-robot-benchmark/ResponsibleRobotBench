# Model access and credentials

[Home](../README.md) · [Installation](installation.md) · [Evaluation](evaluation.md)

The original evaluator's Qwen2-VL backend uses an OpenAI-compatible
`/chat/completions` endpoint. Set the endpoint in the simulator shell and pass
the model name to the evaluator:

```bash
export RRB_BASE_URL="http://127.0.0.1:18017/v1"
export RRB_MODEL="Qwen/Qwen2-VL-7B-Instruct"
```

`RRB_MODEL` is a shell variable used in the documented commands; the evaluator
receives its value through `--model_name`. The client reads `RRB_BASE_URL` when
Python starts. If `remote_url` is also set, it takes precedence. Supply the base
URL without `/chat/completions`.

Set `RRB_API_KEY` privately when the Qwen2-VL endpoint requires authentication;
the backend uses `EMPTY` when it is unset. Other model branches use their own
settings: GPT uses `OPENAI_API_KEY`, Claude uses `ANTHROPIC_API_KEY`, Gemini
uses `GEMINI_API_KEY`, and the lowercase `qwen` branch uses `DASHSCOPE_API_KEY`.
Select the model name and credentials appropriate to the backend.

If you maintain a local `.env` file, create it yourself and load it before
starting Python; the evaluator does not load it automatically. Do not commit
credentials or put them in prompts, examples, endpoint URLs, or shared logs.
Review model responses, observations, and execution records before sharing.

The optional `tools/serve_qwen.py` service has no authentication. Keep it bound
to loopback and use an SSH tunnel for a remote simulator. It supports CONFIG-11
with five example images plus the current image and checks the complete input
and requested output against the loaded model's context capacity. See the
[installation guide](installation.md#5-configure-model-inference) for deployment.
