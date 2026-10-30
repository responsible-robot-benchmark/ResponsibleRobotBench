# Demonstrations and datasets

Full demonstration datasets, model checkpoints, and evaluation outputs are not
distributed with the source repository. Collect and keep them locally, separate
from the configuration examples and independent evaluation split.

Use the demonstration collector at
[`dataset_generator_NLP.py`](../responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py).
It records authored native motion through `get_demos`, with the complete
`ObservationConfig.set_all(True)` configuration. No model endpoint is needed.
All commands use the Python 3.11 simulator environment and display from
[installation](installation.md), from the repository root in Bash.

[Installation](installation.md) · [Planner evaluation](evaluation.md) ·
[RGB policy](policy.md) · [Custom tasks and robots](extending.md)

## Task names and artifact types

The canonical short IDs and task module names are listed in
[tasks.json](../responsible_robot_bench/configs/tasks.json).

| Artifact | Producer and consumer |
| --- | --- |
| Original demonstration episode | `dataset_generator_NLP.py` writes RGB/depth/mask image folders, `low_dim_obs.pkl`, automatic matched YOLO images and JSON, with replay assets when `save_configs` is enabled. |
| RGB Zarr dataset | `convert_pickle_to_zarr.py` selects RGB and policy fields from complete recordings; policy training reads per-task `train`/`val` directories. |
| Planner examples | Task examples are configured in the example configuration file; visual examples use the matching collected images. Code uses shared LMP module prompts. See [example preparation](evaluation.md#prepare-recorded-examples). |
| Evaluation configuration | `test_config_generator.py` saves scene, waypoint, and `configs.pkl` assets without recording a demonstration; evaluators replay them. |

The collector accepts short IDs, such as `water4`, or full registered
names, such as `water4_plants`. Multiple tasks use a comma-separated `--tasks`
value. Policy conversion, training, and rollout use full names.
[tasks.json](../responsible_robot_bench/configs/tasks.json) is the authoritative mapping.

## Collect one demonstration

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 1 --processes 1 --variations 1 \
  --max_attempts 10 --seed 31 --save_path runs/water4-demos/train
```

This is the same collector used for larger datasets. Its default image size is
128 × 128; `--image_size 128,128` sets it explicitly. All five views retain RGB,
depth, and instance masks. Low-dimensional observations retain robot state,
safety/help fields, camera calibration, and other native observation fields.
Point clouds are enabled during collection; the serializer stores depth
and calibration rather than a separate point-cloud array in each saved pickle.
Keep every modality even when a downstream policy consumes only RGB.

The same command automatically annotates every saved frame in all five views
with YOLO11. Install the detector dependencies described in [installation](installation.md).
The default weights are `yolo11n.pt`; set `RRB_YOLO_WEIGHTS` to an existing local
weights file to avoid a download. `RRB_YOLO_DEVICE` defaults to `cpu` and accepts
an Ultralytics device such as `0` with a CUDA-enabled PyTorch installation
(see [installation](installation.md)). No separate annotation command is required.
Annotation uses the original `form_object_coord_for_input` and
`draw_bounding_boxes` helpers: object coordinates come from the four
matching views (front, left shoulder, right shoulder, wrist) and are projected
into all five calibrated images to match YOLO predictions. Object numbers follow
the task's Skill object IDs across camera views and action modes.
When continuing a dataset, stored annotation identities are checked against
the task object IDs. Older numbering is corrected using the recorded object
names and box geometry, with labels redrawn from the raw RGB. Raw recordings
are preserved; this correction does not rerun detection.

Masks identify objects for coordinate matching. Box extents come from YOLO,
except for the microwave body in heat1 and heat2, where its instance mask
provides the box. Other objects in those tasks continue to use YOLO.

`--save_configs=True` is the default and keeps `configs.pkl`, `task_base.ttm`,
and `waypoint_sets.ttm` with each episode. Keep it enabled for scene replay.
The collector's native task assistance behavior and recorded labels are retained;
there is no replacement `--human-help` mode in this entry point.

### Inspect the result

```text
runs/water4-demos/train/water4_plants/variation0/episodes/episode0/
├── low_dim_obs.pkl
├── front_rgb/0.png ...
├── front_depth/0.png ...
├── front_mask/0.png ...
├── ... RGB, depth, and mask folders for the other four cameras
├── front_rgb_annotated/0.png ...
├── ... sibling *_rgb_annotated folders for the other four cameras
├── bbox_annotations.json
├── configs.pkl
├── task_base.ttm
└── waypoint_sets.ttm
```

Images are saved separately; corresponding image and point-cloud attributes in
`low_dim_obs.pkl` are cleared by the serializer. A pickle alone is not
a complete episode. Preserve the companion directories and configuration files.
The annotation JSON records every frame/view, relative image paths, matched
pixel `xyxy` boxes, confidence, weights, object names, world points and voxel
coordinates. A view with no matched detection retains an empty `boxes` list and
an image without boxes; no boxes are fabricated. The JSON and image counts/sizes
are validated before publication. Annotated images never enter the original RGB
folders, whose file counts are checked by the native demonstration loader.
Inspect recorded safety, help, task outcome, and collection failures separately.
A completed task may still be unsafe. Neither an episode directory's presence
nor a successful conversion proves safe task completion.

## Collect 5, 10, 50, or 100 episodes per task

For example, request a total of ten complete episodes for variation 0:

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 10 --processes 1 --variations 1 \
  --max_attempts 10 --seed 100 --save_path runs/native-water4
```

Set `--episodes_per_task` to `5`, `50`, or `100` as needed. Omit `--tasks` to
select all 23 registered tasks. `--variations 1` collects only variation 0;
the default `--variations -1` collects all available variations. Episode targets
apply to each selected task variation, not to a combined batch.

### Continue an existing collection

Repeat the original command with the same `--save_path` and target count, or
increase `--episodes_per_task`. The count includes existing complete episodes:
if five complete episodes exist, a target of ten requests five more.

Resume checks all five cameras' RGB/depth/mask frames and low-dimensional data,
plus replay assets when `--save_configs=True`. Incomplete directories remain
available for inspection and do not count toward the target. New episode IDs
start after the highest existing ID, so gaps and failed attempts are not
silently overwritten. Data is written directly to `episodes/episodeN/`.
`collection_record.json` marks a completed write; a partial write does not count
toward the target. A failed or interrupted write is retained in `failed_episodes/`.

Resume also validates the annotation JSON and all five annotated image folders.
Complete raw episodes without annotations are annotated automatically from
their saved depth, masks and calibration, even when the episode target has
already been reached. Valid annotations are reused without detector inference
or image writes. Raw PNGs, observations and replay assets remain unchanged.


Raw trajectories are saved before annotation. A detector or annotation
failure causes a nonzero exit, records `annotation_error.json` in the episode
and `annotation_failed` in `collection_status.json`, and preserves the complete
raw episode. It does not trigger another simulation attempt or discard a saved
success. Annotated images are written directly to the episode's
`*_rgb_annotated/` folders. Frames already written remain there if annotation
is interrupted. Fix the reported
annotation error and repeat the same command to
finish annotation and continue toward the target. The status report separates
native collection attempts from annotation results.

`--max_attempts 10` bounds each burst of attempts for a missing episode. The
collector then proceeds to the other selected tasks and returns to unfinished
tasks. By default it retries until every task variation reaches its target.
`--max_retry_rounds 3` limits this to three retry rounds after the initial pass;
`--noretry_failed` runs only the initial pass. `--retry_delay 30` sets the pause
between rounds. Exhausting a configured limit, a worker crash, or a success-data
storage/annotation error produces a nonzero exit instead of reporting completion.

The root `collection_queue.json` records pending tasks and the current round.
Each variation's `collection_status.json` retains ordered attempts across
resumes, including outcomes, seeds and exception chains. A persistent attempt
counter ensures seeded retries use new layouts rather than repeating the same
failed seed after every restart.
Do not infer completeness by counting `low_dim_obs.pkl` files alone. The optional
`--seed` controls collection randomness; omit it for unseeded native collection.
A target count is not a guarantee of successful collection or safe outcomes.

### Failed collection attempts

Successful goal-reaching demonstrations remain in `episodes/episodeN/`.
Unsuccessful native collection attempts are saved separately:

```text
water4_plants/variation0/
├── episodes/episode0/          # Goal completed; observation format
├── failed_episodes/
│   ├── attempt000000/          # An unsuccessful attempt, outside the target count
│   │   ├── failure.json
│   │   ├── low_dim_obs.pkl
│   │   ├── front_rgb/ ...     # Actual frames recorded before failure
│   │   └── ...               # Depth, masks, other cameras, YOLO and replay assets
│   └── attempt000003/
│       └── failure.json       # Initialization failed before any frame was recorded
└── collection_status.json
```

Failed trajectories use the observation writer and YOLO annotation
path. `failure.json` records the failure cause, seed, frame count, whether the
trajectory was interrupted, and any missing replay assets or failed annotation.
If initialization fails before observations exist, only metadata is saved.
Failure data does not enter the successful-episode inventory or the default
policy converter's `episodes` selection. Safety labels remain independent of
goal completion, including for inherently unsafe tasks.

These separate recordings can be selected for failure analysis or future policy
learning. Keep their outcomes and partial-trajectory flags with the data when
building such a dataset.

Use separate roots and seeds for training and validation, for example:

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 100 --processes 1 --variations 1 \
  --max_attempts 10 --seed 1000 --save_path runs/il-train
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/dataset_generator_NLP.py \
  --tasks water4 --episodes_per_task 10 --processes 1 --variations 1 \
  --max_attempts 10 --seed 10000 --save_path runs/il-val
```

Inspect the actual recordings and provenance for split overlap; different paths
alone do not prove distinct scenes. Keep incomplete episodes outside the curated
selection passed to conversion, while preserving their original directories.
See [policy data selection](policy.md#prepare-100-demonstrations-per-task).

## Convert RGB demonstrations to Zarr

Install the [policy environment](policy.md#install-policy-dependencies) first.
RGB conversion is an explicit downstream selection from the full dataset; it
does not change collection or remove source modalities. Check that the chosen episode directories contain only complete, intended
examples. Follow the [policy workflow](policy.md) for conversion and training.

Pass the `episodes` directory, or a curated directory of episode symlinks, rather
than the collection root. The converter reads `episode*/low_dim_obs.pkl` and loads
RGB from `<episode>/<camera>_rgb/<frame-index>.png`. Instructions come from the
demo or adjacent `configs.pkl`. It does not recursively discover arbitrary trees,
automatically split episodes, or filter task outcomes. Review the selection before
conversion, retaining original safety labels and provenance.

Existing trusted `demo.pkl` files are also readable by the converter. Policy
conversion does not populate the evaluator's prompt examples. See
[planner examples](evaluation.md#prepare-recorded-examples) for the separate
example configuration workflow.

### Converted schema

Each output directory contains the Zarr replay buffer, `conversion.json`, and
`meta/task_instructions.json`. With the default 128 × 128 image size:

| Array | Shape and meaning |
| --- | --- |
| `images` | `(frames, 5, 128, 128, 3)`, float32 RGB in [0, 1]. Camera order: front, left shoulder, right shoulder, overhead, wrist. |
| `robot_state` | `(frames, 10)`: XYZ, six rotation values from the first two transform columns, and gripper openness. |
| `is_safe` | `(frames, 1)`, recorded binary safety label. |
| `call_human_help` | `(frames, 1)`, recorded binary help flag. |
| `human_intervention_value` | `(frames, 1)`, intervention value from observations or available episode metadata. |

The buffer stores episode boundaries; the JSON metadata records instructions,
frame counts, and source pickle digests. Conversion retains every frame, resizes
RGB, and rejects missing instructions, invalid transforms, nonfinite arrays, or
missing/invalid required safety and help flags. Intervention metadata has legacy
fallbacks; inspect its provenance when mixing older data.

The converter writes directly into a new destination. Existing destinations are
refused. A destination lock excludes concurrent converters; the lock file remains
after completion. On a handled failure, the converter removes only the incomplete
directory created by that invocation. After an unhandled process interruption,
inspect and remove the incomplete destination before retrying. Source recordings
are never removed or overwritten.

Continue with [training and rollout](policy.md#train-and-resume-separately).
Keep the five-camera, 128 × 128 converted layout for the documented RGB models;
changing image size also requires a compatible policy configuration.

## Evaluation configurations

Use independent saved configurations for evaluation. Demonstration episodes
with `--save_configs=True` already include all three replay assets. To create
scene configurations without recording demonstration trajectories, run the
scene-configuration generator:

```bash
.venv/bin/python \
  responsible_robot_bench/envs/rrb_manipulation/tools/test_config_generator.py \
  --tasks water4 --episodes_per_task 5 --episodes_per_task_all_variations 0 \
  --processes 1 --variations 1 --save_path runs/eval-scenes
```

`--episodes_per_task_all_variations` defaults to 100 and overrides
`--episodes_per_task` when positive; set it to `0` as above to request five
per selected variation. The generator seeds itself randomly and does not expose
`--seed`. Use a fresh output root for an independent split, inspect its console
output, and verify the number and contents of saved episodes. Unlike the
resumable demonstration collector, this script counts existing directory entries
and does not provide a collection manifest.

```text
runs/eval-scenes/water4_plants/variation0/episodes/episode0/
├── task_base.ttm
├── waypoint_sets.ttm
└── configs.pkl
```

For this dataset pass `--data_root runs/eval-scenes` to the evaluator.
Use `--selected_indexes 0` for the first saved scene, `--selected_indexes 0,2`
for a subset, or `--selected_indexes ""` for all five. The Code CLI defaults to
`[0]`; explicitly pass `--selected_indexes ""` to run all scenes. See
[evaluation commands](evaluation.md#run-one-episode). Check that the selected
scenes have all replay assets. Distinct paths do not prove distinct scene
contents. For 50 or 100 scenes, collect that count in an independent root and
preserve the assets across comparisons. Collection does not establish planner
success.
