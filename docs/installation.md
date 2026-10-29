# Installation

Set up Python 3.11 and CoppeliaSim for the
[original demonstration collector](dataset.md#collect-one-demonstration) and
[evaluation scene collection](dataset.md#evaluation-configurations). Add a model
service for [original planner evaluation](evaluation.md), and use the separate
[policy environment](policy.md) for downstream RGB conversion and training.
Run all commands from the repository root in **Bash on Linux**.

[Evaluation](evaluation.md) · [Dataset](dataset.md) · [RGB policy](policy.md) ·
[Troubleshooting](#troubleshooting)

## 1. Prepare Python 3.11

You need Python 3.11 with `venv` support and development headers, a C compiler,
and Git. Simulator rendering also needs Xvfb and the system libraries required
by CoppeliaSim's Qt/OpenGL runtime. Keep this source checkout: the `rrb` package
alone does not install simulator scenes or policy resources.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
```

## 2. Install CoppeliaSim and PyRep

Download and verify the CoppeliaSim 4.1 EDU archive:

```bash
mkdir -p .runtime/downloads
curl --fail --location -A 'Mozilla/5.0' \
  -o .runtime/downloads/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
  https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
sha256sum --check <<'SHA256'
512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8  .runtime/downloads/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
SHA256
tar -xf .runtime/downloads/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C .runtime
```

Install the supplied simulator requirements and the additional collector,
detector, and evaluator dependencies, then build the bundled PyRep:

```bash
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-legacy.txt hydra-core omegaconf pyyaml
export COPPELIASIM_ROOT="$PWD/.runtime/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM=xcb
export PYTHONPATH="$PWD:$PWD/responsible_robot_bench/envs/rrb_manipulation${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m pip install --no-build-isolation --no-deps ./thirdparty/PyRep
cp responsible_robot_bench/envs/rrb_manipulation/simAddOnScript_PyRep.lua "$COPPELIASIM_ROOT/"
.venv/bin/python -m pip check
```

`requirements-legacy.txt` includes the simulator, detector, and evaluator
requirements, keeping NumPy and OpenCV compatible. Hydra and OmegaConf support
the original configuration entry point. Install platform-specific Qt/OpenGL and
PyTorch packages for the chosen environment and verify the imports before use.
The exports above configure the simulator library and Python import paths;
repeat them from the repository root in each simulator shell. The PyRep build
needs write access to the simulator's `system/` directory. YOLO11 annotates collected
frames and CONFIG-9 observations; `RRB_YOLO_WEIGHTS` can select a local file.

Component terms and attribution are listed in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Consult the simulator's own
terms when choosing an installation.

## 3. Configure a display

CoppeliaSim needs a working X11 display, including in headless simulator mode.
Use an existing display or start Xvfb on an available display number, for
example:

```bash
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
export DISPLAY=:99
```

Choose an unused display and ensure the Qt/OpenGL libraries required by the
simulator are installed. Software rendering can be selected with
`export LIBGL_ALWAYS_SOFTWARE=1` when supported by the host.

## 4. Check collection

With the simulator environment and display configured, use the original
[one-episode collector command](dataset.md#collect-one-demonstration). Inspect
the saved RGB/depth/mask folders, low-dimensional observations, replay assets,
and collection logs. One collected episode does not validate the full task suite,
model inference, or policy training.

## 5. Configure model inference

The simulator environment must also contain the evaluator dependencies listed
in step 2. Confirm its imports before running Skill, Pose, or Code evaluation.

The documented Qwen2-VL backend uses the original `RemoteModel` with the OpenAI
client to call an OpenAI-compatible `/chat/completions` endpoint. Text CONFIGs
need text completion; RGB CONFIGs also need inline image support. The model
service can use a separate environment or host from the simulator.

### Connect to a service

In the simulator shell:

```bash
export RRB_BASE_URL="http://127.0.0.1:18017/v1"
export RRB_MODEL="Qwen/Qwen2-VL-7B-Instruct"
```

| Variable | Meaning |
| --- | --- |
| `RRB_BASE_URL` | API base URL, including `/v1`; the client appends `/chat/completions`. |
| `RRB_MODEL` | Model ID sent to the endpoint. |
| `RRB_API_KEY` | Passed to the original Qwen2-VL `RemoteModel` client. Set it in your local environment when the endpoint requires authentication. |

Pass the model ID to the original evaluators as `--model_name "$RRB_MODEL"`.
For Qwen2-VL, set `RRB_BASE_URL` before starting Python; the original client
also accepts `remote_url`, which takes precedence when set. Supply the API base
URL without the `/chat/completions` suffix. Other model backends retain their
own endpoint and credential settings.

Use the original
[example selectors and module prompts](evaluation.md#prepare-recorded-examples).

Use a compatible model service appropriate for your selected model. RGB
configurations need inline image support; CONFIG-11 needs the current image and
five matching example images. The optional local `tools/serve_qwen.py` supports
these six images, the original JSON-schema requests, and completion stop strings.
It preserves the full request and checks input plus requested output tokens
against the loaded model's context capacity. It does not impose a four-image or
2048-output-token cap.

To host Qwen2-VL-7B on a CUDA 12.4-compatible system:

```bash
python3.11 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv-qwen/bin/python -m pip install -r requirements-model.txt
.venv-qwen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-7B-Instruct', local_dir='models/Qwen2-VL-7B-Instruct')"
export RRB_MODEL_PATH="$(pwd)/models/Qwen2-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 .venv-qwen/bin/python -m uvicorn tools.serve_qwen:app \
  --host 127.0.0.1 --port 18017 --workers 1
```

Keep the service running and invoke the evaluators in another shell. For another
CUDA version, select a matching PyTorch build. Use loopback or an SSH tunnel;
follow the provider's authentication and network guidance.

Continue with [evaluation scene collection](dataset.md#evaluation-configurations),
[original planner evaluation](evaluation.md#run-one-episode), and
[100-demonstration policy training](policy.md#prepare-100-demonstrations-per-task).
For new environments, see [custom tasks and robots](extending.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `COPPELIASIM_ROOT not defined` or missing simulator libraries | Set the simulator exports above in the current Bash shell and check the extracted directory. |
| Xvfb executable is missing | Install Xvfb or use an existing X11 display. |
| Qt/OpenGL context creation fails | Check `DISPLAY`, the Qt plugin path, and the host's X11/OpenGL libraries. |
| `FileExistsError` for a run directory | Use separate directories for independent splits and runs. The original demonstration collector continues complete episodes in the same `--save_path`; see [collection resume](dataset.md#continue-an-existing-collection). |
| Model connection or authentication error | Check the base URL, model ID, service health, and any endpoint-required token. A base URL must not already end in `/chat/completions`. |
| Context-limit rejection | Check the service context limit and request size. Keep the requested CONFIG's examples intact; changing the shot count changes the experiment. |
| NumPy or Hugging Face dependency conflicts | Keep simulator, model, and policy environments separate. Simulator package pins do not apply to the policy environment. |
