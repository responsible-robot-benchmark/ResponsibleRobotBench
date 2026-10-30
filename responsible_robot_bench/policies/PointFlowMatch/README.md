# PointFlowMatch for ResponsibleRobotBench

Train image-based flow-matching policies by imitation learning from collected
ResponsibleRobotBench demonstrations. The policy uses five RGB camera views and
robot state. Each training sample contains three items: observation images,
observed robot states, and future robot states. The action has ten components:
position (3), rotation represented by two matrix columns (6), and gripper (1).

## Installation

Use Python 3.11 and the dependencies in [the policy guide](../../../docs/policy.md).
Run the commands below from the repository root. The existing training entry
points are `scripts/train.py` and `scripts/train_vlm.py`. Both use Hydra.

## Collect and convert demonstrations

Collect data as described in [the dataset guide](../../../docs/dataset.md), retaining
the five camera folders and `low_dim_obs.pkl`. Keep training and validation
episodes separate from prompt examples and evaluation episodes.

Convert each split with the existing converter:

```bash
python responsible_robot_bench/policies/PointFlowMatch/scripts/convert_pickle_to_zarr.py \
  --data_root datasets/train/water4_plants/variation0/episodes \
  --task_name water4_plants --output_path datasets/policy/water4_plants/train \
  --split train
python responsible_robot_bench/policies/PointFlowMatch/scripts/convert_pickle_to_zarr.py \
  --data_root datasets/valid/water4_plants/variation0/episodes \
  --task_name water4_plants --output_path datasets/policy/water4_plants/val \
  --split val
```

Embedded `demo.pkl` episodes are also supported. The converter preserves the
source files, writes RGB as float32 in [0, 1], and rejects nonfinite motion data.
It writes directly into a new destination under a conversion lock and refuses
existing files, directories, or symlinks. A caught failure removes only its own
partial output. An uncatchable interruption can leave partial output, which must
be inspected before retrying. The lock file is retained for concurrent-writer
safety; no staging directory is used.

Recorded descriptions and per-frame safety/help annotations are retained in the
episode metadata in `conversion.json`. Missing annotations are not invented.
These annotations are not model inputs or additional prediction dimensions.

## Train

```bash
python responsible_robot_bench/policies/PointFlowMatch/scripts/train_vlm.py \
  +experiment=pointflowmatch_images task_name=water4_plants \
  "data_root=$PWD/datasets/policy" "checkpoint_dir=$PWD/checkpoints"
```

The image model uses five ResNet18 encoders and the original conditional
flow-matching U-Net. The observation dimension is 2570 per time step
(5 × 512 image features + 10 robot-state components); two observations give
5140 conditioning features. Both native and converted RGB readers return the
same three-item sample with RGB normalized to [0, 1].

Edit `conf/train.yaml` and `conf/experiment/` for training settings:
- `pointflowmatch_images`: converted RGB demonstrations, 32 prediction steps.
- `water2_config`: converted RGB demonstrations, 64 prediction steps.
- `vlmbench_base`: native `<data_root>/{train,valid}/<task>/variation*/episodes`
  folders, 8 prediction steps.

The converted reader accepts `val` or the original `valid` split name. Task
lists select joint training. All training and validation batches are used by
default. Existing subset settings can be explicitly overridden for short checks.

For native folders, use `+experiment=vlmbench_base data_root=datasets/native`.
For continuation, set `load_path` to a compatible checkpoint. The effective
configuration is saved with checkpoints; `run_name` names an output run and
does not restore weights on its own.

## Evaluate

Use `scripts/eval_vlmbench.py --ckpt_dir <checkpoint-directory>` with the options
documented in [the policy guide](../../../docs/policy.md). Evaluation calls the image
policy with RGB and robot state, executes ten-component actions, and records
actual task success, safety, and action counts from the simulator. A successful
training or loading check does not establish task success.

## Attribution

This integration is based on **Learning Robotic Manipulation Policies from
Point Clouds with Conditional Flow Matching**, by Eugenio Chisari, Nick Heppert,
Max Argus, Tim Welschehold, Thomas Brox, and Abhinav Valada (CoRL 2024).
See the [PointFlowMatch project](http://pointflowmatch.cs.uni-freiburg.de/),
the original [GPL v3 license](LICENSE), and
[component license files](../../../licenses/).
