# RGB policy workflow

Train PointFlowMatch (PFM) from inspected native RGB demonstrations, then
restore a checkpoint and evaluate it in CoppeliaSim. The workflow below prepares
100 training demonstrations per selected task with a separate validation split.
All commands below use **Python 3.11** and run from the repository root in Bash.

[Installation](installation.md) · [Dataset](dataset.md) ·
[Planner evaluation](evaluation.md) · [Extensions](extending.md)

## Inputs

Prepare native motion trajectories with [the dataset workflow](dataset.md).
Planner prompt examples require their own interface-specific validation.

This guide uses the image flow-matching policy: five RGB cameras and
observed robot states condition ten-dimensional targets (position, rotation-6D,
and gripper). Each training sample contains images, observed states, and future
states. Safety and assistance fields remain dataset metadata; they are not
additional policy inputs or prediction targets. The collector records all
modalities. Preserve the source depth and masks for other uses. Simulator
checkers determine task and safety outcomes during evaluation.

## Install policy dependencies

First complete [simulator installation](installation.md). Use a separate
`.venv-ml`: the policy pins NumPy 1.23.5, while the simulator requirements pin
1.26.4. The policy also pins an older Hugging Face stack than the Qwen server.

```bash
python3.11 -m venv .venv-ml
.venv-ml/bin/python -m pip install --upgrade pip setuptools wheel
.venv-ml/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu
.venv-ml/bin/python -m pip install -e ./responsible_robot_bench/policies/PointFlowMatch numcodecs==0.13.1
.venv-ml/bin/python -m pip install -e . cffi pyquaternion==0.9.9 \
  natsort==8.4.0 transforms3d gymnasium==1.4.0 plotly==7.1.0
```

Install the pinned Diffusion Policy dependency with the supplied packaging fix:

```bash
mkdir -p .runtime/deps
git clone https://github.com/chisarie/diffusion_policy.git .runtime/deps/diffusion_policy
git -C .runtime/deps/diffusion_policy checkout a21dd5cb94dcff5495cb48bb7e3d59cb0b94c014
git -C .runtime/deps/diffusion_policy apply --check "$(pwd)/responsible_robot_bench/policies/PointFlowMatch/patches/diffusion-policy-packaging.patch"
git -C .runtime/deps/diffusion_policy apply "$(pwd)/responsible_robot_bench/policies/PointFlowMatch/patches/diffusion-policy-packaging.patch"
.venv-ml/bin/python -m pip install --no-deps .runtime/deps/diffusion_policy

source tools/sim_env.sh
.venv-ml/bin/python -m pip install --no-build-isolation --no-deps ./thirdparty/PyRep
.venv-ml/bin/python -m pip check
```

Run the clone and patch steps once in a fresh dependency directory. The patch
includes upstream namespace packages in the installed wheel; it does not change
policy behavior. Component attribution is in
[component license files](../licenses/).

The recipe above uses CPU PyTorch. Policy dependencies are specified in
[pyproject.toml](../responsible_robot_bench/policies/PointFlowMatch/pyproject.toml). Keep
`numcodecs==0.13.1` with the
retained Zarr stack and the declared `chardet>=3.0.2,<6` bound for Requests
compatibility.

The image policy uses the ResNet backbone selected by
`+experiment=pointflowmatch_images`. Keep the same backbone and model configuration
when restoring checkpoints.

## Prepare 100 demonstrations per task

Use the [collector](dataset.md#collect-5-10-50-or-100-episodes-per-task)
to request 100 complete training episodes in `runs/il-train` and 10 validation
episodes in `runs/il-val`, with separate collection seeds. The target counts
include existing complete episodes when continuing collection. Inspect collection
logs, `collection_status.json`, and native outcome/safety fields; missing or incomplete episodes do not
satisfy the requested count.

For `water4`, the episode directories are:

```bash
export RRB_IL_TRAIN="runs/il-train/water4_plants/variation0/episodes"
export RRB_IL_VAL="runs/il-val/water4_plants/variation0/episodes"
```

Keep each episode's RGB/depth/mask folders, `low_dim_obs.pkl`, and replay assets
together. If either directory also contains incomplete or unwanted episodes,
create a separate curated directory of `episodeN` symlinks to the accepted source
directories and point the corresponding variable at that directory. Do not delete
failed recordings to make the converter's input look complete.

Record the selected paths, collection seeds, native outcome labels, and selection
criterion with the curated dataset. Check that training and validation contain
100 and 10 accepted episodes respectively, and check for duplicated source scenes.
Different directory names alone do not establish disjoint data. Retain original
safety labels, including unsafe outcomes. If the experiment requires safe task
completion, explicitly select for both safety and task completion; inherently
unsafe goals cannot supply safe, successful demonstrations.

## Convert existing demonstrations

After installing policy dependencies, convert the selected data:

```bash
.venv-ml/bin/python responsible_robot_bench/policies/PointFlowMatch/scripts/convert_pickle_to_zarr.py \
  --data_root "$RRB_IL_TRAIN" --task_name water4_plants \
  --output_path runs/il-data/water4_plants/train --split train
.venv-ml/bin/python responsible_robot_bench/policies/PointFlowMatch/scripts/convert_pickle_to_zarr.py \
  --data_root "$RRB_IL_VAL" --task_name water4_plants \
  --output_path runs/il-data/water4_plants/val --split val
```

Repeat conversion for each selected task using its full registered name.
[Dataset formats](dataset.md#convert-rgb-demonstrations-to-zarr) explain the
five-camera schema and output handling. Inspect each `conversion.json`
for `n_episodes`, frame counts, source digests, and terminal labels. The converter
accepts a flat directory of episode children; pass the native `episodes` directory
or the curated links, not the collection root containing all tasks. It does not
filter collection failures or infer missing task-success labels. Original
collection progress records retain the task outcomes; keep them with the data.

Policy scripts use full names such as `water4_plants`; collector and planner
CLIs use short IDs such as `water4`. Training expects
`runs/il-data/<full-task-name>/train` and `val`.

## Train and resume separately

With 100 selected training demonstrations and the validation split converted,
train over the complete dataset. The following selects RGB explicitly and retains
the dimensions from `conf/train.yaml`, `conf/model/flow.yaml`, and
`conf/backbone/resnet_dp.yaml`: ten-dimensional actions, 32 prediction steps,
and 512 image features per camera. It uses all batches, ten
epochs, and a checkpoint each epoch. Choose the duration for your experiment;
these commands do not establish policy performance.

The CPU environment matches the installation recipe above. For GPU training,
install a matching CUDA PyTorch build in the policy environment. All PFM sources and
experiment configurations remain available; this RGB workflow does not validate
other backbones or point-cloud experiments.

```bash
export RRB_POLICY_DATA="$(pwd)/runs/il-data"
export RRB_POLICY_RUN="$(pwd)/runs/pfm-water4-il100"
.venv-ml/bin/python responsible_robot_bench/policies/PointFlowMatch/scripts/train.py \
  task_name=water4_plants "data_root=$RRB_POLICY_DATA" \
  "checkpoint_dir=$RRB_POLICY_RUN/checkpoints" \
  +experiment=pointflowmatch_images obs_mode=rgb seed=31 run_name=train \
  max_duration=10ep save_interval=1ep \
  train_subset_num_batches=-1 eval_subset_num_batches=-1 \
  "hydra.run.dir=$RRB_POLICY_RUN/hydra-train"

.venv-ml/bin/python responsible_robot_bench/policies/PointFlowMatch/scripts/train_vlm.py \
  task_name=water4_plants "data_root=$RRB_POLICY_DATA" \
  "checkpoint_dir=$RRB_POLICY_RUN/checkpoints" \
  +experiment=pointflowmatch_images obs_mode=rgb seed=31 run_name=resume \
  max_duration=20ep save_interval=1ep \
  train_subset_num_batches=-1 eval_subset_num_batches=-1 \
  "load_path=$RRB_POLICY_RUN/checkpoints/train/latest-rank0.pt" \
  "hydra.run.dir=$RRB_POLICY_RUN/hydra-resume"
```

Both scripts call the same training implementation. They use Hydra `key=value`
overrides, not argparse flags. `load_path` is required for an explicit resume;
`run_name` alone does not load a checkpoint. `max_duration=20ep` sets the total
training duration, including restored progress. Preserve the model configuration
between training and resume so the checkpoint's parameter shapes remain compatible.
Repeat this command with each task's full name and a separate run directory for
per-task policies. The trainer also accepts a Hydra task list, for example
`'task_name=[water4_plants,water2_plants]'`, for joint training when both tasks'
`train` and `val` directories are present. Keep the saved `config.yaml`
alongside the checkpoint. W&B logging is disabled by default.

## Restore and roll out

With the manual run above, execute the resumed policy:

```bash
source tools/sim_env.sh
.venv-ml/bin/python tools/with_display.py .venv-ml/bin/python \
  responsible_robot_bench/policies/PointFlowMatch/scripts/eval_vlmbench.py \
  --ckpt_dir "$RRB_POLICY_RUN/checkpoints/resume" \
  --task_name water4_plants --ckpt_epoch latest \
  --num_episodes 1 --max_steps 16 --num_k_infer 2 --seed 33 --headless
```

The evaluator restores `config.yaml` and the selected checkpoint;
`latest` prefers `latest-rank0.pt` and otherwise selects by numeric epoch/batch.
`--num_k_infer` controls flow integration steps, not episode count. Headless
rendering still needs the display launcher.

Results appear under `<checkpoint-directory>/eval_results/water4_plants/`:

- `eval_water4_plants_latest.json`: episode labels and rates, expressed as percentages.
- `action_trace_water4_plants_latest.json`: predicted actions, execution feedback,
  observed poses, and terminal labels.
- Timestamped copies of both records; the `latest` files are updated on later evaluations.

In policy `episode_details`, `success` is task completion, `safe` is terminal
safety, and `safety_success` is their joint label. This differs from planner
reports, where `safety_success` means safety alone. Always inspect the schema
before combining results.

For 50 or 100 policy evaluation episodes, set `--num_episodes 50` or
`--num_episodes 100` on the rollout command. Choose and report an appropriate
`--max_steps` budget; the example above uses 16 steps. A completed rollout can
fail its task. Keep episode labels and action traces when reporting results.
