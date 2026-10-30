<div align="center">

<img src="docs/assets/rrb-logo.png" alt="ResponsibleRobotBench task ring" width="300">

# ResponsibleRobotBench

**Recognize hazards. Plan safe actions. Complete manipulation tasks.**

23 task configurations · 3 action interfaces · 12 planner configurations

[Installation](docs/installation.md) · [Data collection](docs/dataset.md) · [Evaluation](docs/evaluation.md) · [Custom tasks & robots](docs/extending.md)

</div>

ResponsibleRobotBench evaluates robotic manipulation under task-specific safety
constraints. It connects hazard judgments, action choices, execution feedback,
and task outcomes through three interfaces: **high-level skills**, **target poses**,
and **robot-API code**. Collected demonstrations also support imitation learning.

<p align="center">
  <img src="docs/assets/benchmark-overview.png" alt="ResponsibleRobotBench: task hazards, three action representations, multimodal planners, and task outcomes" width="100%">
</p>

## Installation

From the root of your source checkout, create the Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Follow the [installation guide](docs/installation.md) to set up CoppeliaSim 4.1,
PyRep, rendering, and the model service. The editable package install alone is
not enough to run simulation. Use Bash on Linux for the commands below.

## 1. Collect task data

Collected demonstrations and evaluation results are not included in this repository.
Use the commands below to create your own local datasets. Bundled task scenes
and prompt-example images are runtime resources, not an evaluation dataset.

Use the demonstration collector, which enables the complete observation
configuration, including all five cameras' RGB, depth, and instance masks:

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 10 --processes 1 --variations 1 \
  --max_attempts 10 --seed 0 --save_path runs/water4-examples
```

`--episodes_per_task` is the total number of complete episodes per task variation,
including existing episodes when continuing collection. Use `10`, `50`, or `100`
to increase that target; `--tasks water4,water2` selects multiple tasks. The
RGB/depth/mask folders, low-dimensional observations, and replay configs
remain together. The same command also saves YOLO11-annotated images and box
metadata; no manual annotation step is needed. Failed attempts are saved separately
and do not count toward the target. The collector retries unfinished tasks by
default; use `--noretry_failed` for a single pass. See
[dataset collection](docs/dataset.md) for resume behavior, failure records, and
downstream data selection.

## 2. Select planner examples

Keep the episodes used to prepare examples separate from the episodes used for
evaluation. For example, collect ten example episodes in
`runs/water4-examples` and collect independent evaluation episodes in
`runs/water4-evaluation`, using the same collector with a separate `--save_path`
and seed. Set `--episodes_per_task` to the number of evaluation episodes needed.
The collected episodes contain `task_base.ttm`, `waypoint_sets.ttm`, and
`configs.pkl` for replay. Do not evaluate on an episode that supplied a prompt
example. Alternatively, use the
[scene-configuration collector](docs/dataset.md#evaluation-configurations) to
prepare a separate evaluation split without demonstration trajectories.

For example, collect 100 evaluation episodes with:

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 100 --processes 1 --variations 1 \
  --max_attempts 10 --seed 1000 --save_path runs/water4-evaluation
```

Use the collected images as visual examples and keep their matching task text
in the configuration described below. Check that the image labels and text
refer to the same objects before selecting them for visual ICL.

Copy the selected images into the fixed configuration folder with the existing
example script:

```bash
source tools/sim_env.sh
.venv/bin/python -m \
  responsible_robot_bench.evaluator.rrb_manipulation_evaluator_with_general_planner_example_generation \
  --copy_example_images --data_root runs/water4-examples \
  --eval_sets water4 --n_shots 10
```

This replaces image positions 1–10 using collected episodes 0–9, preserving the
raw data and image resolution. Historical annotation numbers are corrected
from recorded task identities before copying. It makes no model calls and does not modify
reference answers. See the [example guide](docs/evaluation.md#prepare-recorded-examples)
for selecting other episode indexes and matching their text order.

Write or edit task examples in
[`rrb_manipulation_example.py`](responsible_robot_bench/evaluator/config/rrb_manipulation_example.py).
The evaluator selects same-task or cross-task examples using its four
selectors. Skill examples use action lists, Pose examples use pose lists, and
Code uses the shared planner, composer, and map prompts in
[`prompts_voxposer/`](responsible_robot_bench/evaluator/prompts_voxposer/). CONFIG-11
also uses the task's image and text examples. The
[example guide](docs/evaluation.md#prepare-recorded-examples) explains how to
add examples and pair images; the [CONFIG reference](docs/evaluation.md#choose-a-config)
lists the required counts and inputs.

## 3. Evaluate skills, poses, and code

Choose a model through the [model API configuration](docs/credentials.md):
GPT, Claude, Gemini, DashScope Qwen, and supported local backends use the same
evaluation commands with their corresponding model names and API keys.
The example below uses a local Qwen2-VL endpoint.

```bash
export RRB_BASE_URL="http://127.0.0.1:18017/v1"
export RRB_MODEL="Qwen/Qwen2-VL-7B-Instruct"
```

Evaluate the first independently collected evaluation scene through the evaluators. This
example uses CONFIG-5 (an RGB input without demonstrations):

```bash
.venv/bin/python -m \
  responsible_robot_bench.evaluator.rrb_manipulation_evaluator_with_general_planner \
  --model_name "$RRB_MODEL" --model_type remote --eval_sets water4 --headless 1 \
  --data_root runs/water4-evaluation --selected_indexes 0 \
  --language_only 0 --n_shots 0 --cognition 0 --chat_history 0 \
  --multiview 0 --detection_box 0 --multistep 0 --visual_icl 0 \
  --example_classification_type same_task_with_hazard --traj_type pre-defined --output_dir runs/water4-skills

.venv/bin/python -m \
  responsible_robot_bench.evaluator.rrb_manipulation_evaluator_with_general_planner \
  --model_name "$RRB_MODEL" --model_type remote --eval_sets water4 --headless 1 \
  --data_root runs/water4-evaluation --selected_indexes 0 \
  --language_only 0 --n_shots 0 --cognition 0 --chat_history 0 \
  --multiview 0 --detection_box 0 --multistep 0 --visual_icl 0 \
  --example_classification_type same_task_with_hazard --traj_type point --output_dir runs/water4-poses

.venv/bin/python -m \
  responsible_robot_bench.evaluator.rrb_manipulation_evaluator_with_general_planner_code_generation \
  --model_name "$RRB_MODEL" --model_type remote --eval_sets water4 --headless 1 \
  --data_root runs/water4-evaluation --selected_indexes 0 \
  --language_only 0 --n_shots 0 --cognition 0 --chat_history 0 \
  --multiview 0 --detection_box 0 --multistep 0 --visual_icl 0 \
  --example_classification_type same_task_with_hazard --traj_type code --output_dir runs/water4-code
```

`--selected_indexes 0` runs the first scene. Set `--selected_indexes ""` to
evaluate all available scenes. The Code CLI defaults to the first scene; pass
`--selected_indexes ""` explicitly to run all scenes. The compatibility option
`--expected_episodes` does not limit or validate the dataset size. The
[evaluation guide](docs/evaluation.md#choose-a-config)
explains the flags for CONFIG-0–11, examples, and result records. Skills and
poses execute action lists; Code retains task planning, composition, object
queries, spatial maps, motion planning, execution, and feedback.

## 4. Train an imitation policy

For imitation learning, collect 100 demonstrations per selected task, prepare
separate training and evaluation data, and train a task-specific policy. For
example, use `--episodes_per_task 100` when collecting `water4`.

The [PFM workflow](docs/policy.md) covers explicit RGB selection from the full
recordings, conversion, training, checkpoint restoration, and simulator evaluation.
The complete PFM sources and configurations are available in `responsible_robot_bench/policies/PointFlowMatch/`.

## 5. Add your own tasks or robot

The [extension guide](docs/extending.md) walks through:

- Building a task scene, randomized layouts, and demonstration waypoints.
- Defining task completion, safety checks, and available assistance.
- Registering the task and preparing descriptions and planner examples.
- Adapting observations and action execution to a robot interface.

## Real-robot examples

<table>
  <tr>
    <th align="center">Franka · Imitation learning</th>
    <th align="center">UR5 · Task and motion planning</th>
  </tr>
  <tr>
    <td align="center"><a href="docs/assets/franka-watering.mp4"><img src="docs/assets/franka-watering.jpg" height="220" alt="Franka moves the power strip before watering"></a></td>
    <td align="center"><a href="docs/assets/ur5-watering.mp4"><img src="docs/assets/ur5-watering.jpg" height="220" alt="UR5 relocates the power strip before watering"></a></td>
  </tr>
  <tr>
    <td align="center"><a href="docs/assets/franka-watering.mp4">▶ View Franka video</a></td>
    <td align="center"><a href="docs/assets/ur5-watering.mp4">▶ View UR5 video</a></td>
  </tr>
</table>

Both examples relocate the power strip before watering the plant.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Environment, simulator, rendering, and model server |
| [Data collection](docs/dataset.md) | Demonstrations, batch collection, and dataset conversion |
| [Planner evaluation](docs/evaluation.md) | Three interfaces, CONFIG-0–11, batch runs, metrics, and logs |
| [Error types](docs/errors.md) | Failure IDs, repeated events, timeouts, and diagnostic coverage |
| [Imitation learning](docs/policy.md) | PFM data, training, checkpoints, and rollout |
| [Custom tasks & robots](docs/extending.md) | Scene creation, registration, and execution adapters |
| [API configuration](docs/credentials.md) | Endpoint settings and private credentials |

## Code structure

```text
responsible_robot_bench/
  configs/               Task registry, CONFIG-0–11 switches, and Hydra settings
  evaluator/             Skill, Pose, and Code evaluation; prompt examples and results
  planner/               Model APIs, prompt assembly, and response parsing
  envs/rrb_manipulation/  Simulator, tasks, scenes, and motion controllers
  data/                  Demonstration storage, annotation, and resume handling
  perception/            Observations, bounding boxes, and object numbering
  execution/             Action feedback, costs, and human assistance
  diagnostics/           Repetition, wrong-target, and collision diagnostics
  policies/PointFlowMatch/  Imitation-learning data conversion, training, and rollout
thirdparty/PyRep/         Simulator binding
tools/                   Model serving, rendering, and environment setup
docs/                    User guides and project media
```

## Acknowledgements

ResponsibleRobotBench builds on the manipulation infrastructure of
[EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench),
[RLBench](https://github.com/stepjam/RLBench), and
[PyRep](https://github.com/stepjam/PyRep). Code generation builds on
[VoxPoser](https://github.com/huangwl18/VoxPoser), including its LMP composition,
spatial maps, and motion-planning components. The imitation-learning integration uses
[PointFlowMatch](https://github.com/robot-learning-freiburg/PointFlowMatch).
We thank the authors and contributors of these projects for sharing their code,
tools, and research.

## License

Original ResponsibleRobotBench code and documentation use the [MIT License](LICENSE).
Included components retain their own license files and source attributions.
