'VLMBench strategy evaluation script - Evaluate trained strategy in simulation environment\npython scripts/eval_vlmbench.py --ckpt_dir ckpt/1770036236-translucent-baboon/ --ckpt_epoch ep900 --max_steps 200 --instruction_str \'["water plant"]\''
import os
import sys
import numpy as np
from pathlib import Path
from omegaconf import ListConfig

# Add necessary paths
# 1. Try to get diffusion_policy path from environment variable
_diffusion_policy_path = os.environ.get('DIFFUSION_POLICY_PATH', None)

# 2. If environment variable does not exist, attempt automatic detection
_project_root = Path(__file__).resolve().parents[4]
_rrb_manipulation = _project_root / 'responsible_robot_bench' / 'envs' / 'rrb_manipulation'

if _diffusion_policy_path is None:
    _possible_paths = [
        _project_root.parent / 'diffusion_policy',
        Path.home() / 'Desktop' / 'diffusion_policy',
    ]
    for _path in _possible_paths:
        if _path.exists() and ((_path / 'diffusion_policy').exists() or (_path / 'model').exists()):
            _diffusion_policy_path = str(_path)
            break

_paths_to_add = [str(_project_root), str(_rrb_manipulation)]
if _diffusion_policy_path is not None:
    _paths_to_add.insert(0, _diffusion_policy_path)

for _path in _paths_to_add:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import argparse
import torch
import wandb
import hydra
import json
from datetime import datetime
from omegaconf import OmegaConf

from pfp.policy.base_policy import BasePolicy
from pfp.envs.vlmbench_runner import VLMBenchRunner


from pfp.evaluation import summarize_episodes

def main():
    parser = argparse.ArgumentParser(description='Evaluate a VLMBench policy')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Checkpoint directory')
    parser.add_argument('--ckpt_epoch', type=str, default='latest',
                        help='Checkpoint epoch, e.g. ep0, ep100, latest')
    parser.add_argument('--task_name', type=str, default=None,
                        help='Task name; read from the configuration when omitted')
    parser.add_argument('--n', type=int, default=None,
                        help='Select task index n (zero-based) from a task list')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--num_k_infer', type=int, default=50, help='Flow integration steps')
    parser.add_argument('--seed',type=int,default=0,help='Simulator and policy random seed')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--variation', type=int, default=0,
                        help='Task variation')
    parser.add_argument('--headless', action='store_true',
                        help='Run without a graphical window')
    parser.add_argument('--log_wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed diagnostics')
    parser.add_argument('--instruction_str', type=json.loads, default=None,
                        help='Task description recorded with evaluation results; image FM has no language input')

    args = parser.parse_args()
    if min(args.num_episodes,args.max_steps,args.num_k_infer)<1:
        parser.error('num_episodes, max_steps and num_k_infer must be positive')
    from pfp import set_seeds
    set_seeds(args.seed)

    if args.instruction_str is not None and not isinstance(args.instruction_str, list):
        args.instruction_str = [args.instruction_str]

    # Load configuration
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        # Try relative path
        ckpt_dir = Path(__file__).parent.parent / 'ckpt' / args.ckpt_dir

    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {args.ckpt_dir}")

    config_path = ckpt_dir / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Configuration file does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    print(f"Configuration loaded: {config_path}")

    # Determine task name
    task_name = args.task_name
    if task_name is None:
        # Read from configuration
        if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'task_name'):
            task_name = cfg.experiment.task_name
        elif hasattr(cfg, 'task_name'):
            task_name = cfg.task_name
        else:
            raise ValueError("Cannot determine task name; specify --task_name")

    print(f"Task: {task_name}")

    config_task_count = None
    task_indices_for_eval = []
    if isinstance(task_name, ListConfig):
        task_name = list(task_name)
        config_task_count = len(task_name)
        print(f"Configured task count: {len(task_name)}")
        if args.n is None:
            raise ValueError("A task list requires --n to select a zero-based task index")
        if args.n < 0 or args.n >= len(task_name):
            raise ValueError(f"Task index out of range: n={args.n}, valid range [0, {len(task_name)-1}]")
        selected_task = task_name[args.n]
        task_name = [selected_task]
        task_indices_for_eval = [args.n]
        print(f"Selected task index n={args.n}: {selected_task}")
    else:
        task_indices_for_eval = [args.n if args.n is not None else None]

    def _get_runtime_overrides(loop_idx):
        return (args.instruction_str[loop_idx % len(args.instruction_str)]
                if args.instruction_str else None)

    from pfp.checkpoints import resolve_checkpoint
    ckpt_path = resolve_checkpoint(ckpt_dir, args.ckpt_epoch)

    print(f"Checkpoint loaded: {ckpt_path}")

    # Load model (use load_from_checkpoint method)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    # Determine model class
    if hasattr(cfg, 'experiment') and hasattr(cfg.experiment, 'model'):
        model_cfg = cfg.experiment.model
    else:
        model_cfg = cfg.model

    model_class_name = model_cfg._target_.split(".")[-1]
    model_module_path = model_cfg._target_.rsplit(".", 1)[0]
    model_module = __import__(model_module_path, fromlist=[model_class_name])
    model_class = getattr(model_module, model_class_name)

    # Load model using load_from_checkpoint method
    # Determine checkpoint episode name
    if args.ckpt_epoch == 'latest':
        ckpt_episode = ckpt_path.stem.split("-")[0]  # Extract from file name, e.g. ep2000-ba92000-rank0.pt -> ep2000
    else:
        ckpt_episode = args.ckpt_epoch if not args.ckpt_epoch.endswith('.pt') else args.ckpt_epoch[:-3]
        # Remove -rank0 suffix (if present)
        if ckpt_episode.endswith('-rank0'):
            ckpt_episode = ckpt_episode[:-6]

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Model: {model_class_name}")
    print(f"✓ Checkpoint episode: {ckpt_episode}")

    # Use the load_from_checkpoint method of the model
    policy: BasePolicy = hydra.utils.instantiate(model_cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(checkpoint['state']['model'], strict=True)
    # Release the duplicate checkpoint tensors before starting the simulator.
    del checkpoint
    policy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
    policy.set_num_k_infer(args.num_k_infer)

    print(f"Model loading completed")

    # Create an environment runner
    if isinstance(task_name, (ListConfig, list)):
        for i, task in enumerate(task_name):
            instruction_override = _get_runtime_overrides(i)
            wandb.init(
                project="vlmbench-eval",
                name=f"{task}_{ckpt_dir.name}_{args.ckpt_epoch}",
                config={
                    "task_name": task,
                    "ckpt_dir": str(ckpt_dir),
                    "ckpt_epoch": args.ckpt_epoch,
                    "num_episodes": args.num_episodes,
                    "seed": args.seed,
                    "variation": args.variation,
                    "task_index": task_indices_for_eval[i] if i < len(task_indices_for_eval) else None,
                    "instruction": instruction_override,
                },
                mode="online" if args.log_wandb else "disabled"
            )

            env_runner = VLMBenchRunner(
                task_name=task,
                num_episodes=args.num_episodes,
                max_episode_length=args.max_steps,
                image_size=(128, 128),
                headless=args.headless,
                variation=args.variation,
                verbose=args.verbose,
                instruction=instruction_override,
            )

            # Run evaluation
            print(f"\nStarting evaluation of task {i+1}/{len(task_name)}: {task}, instruction={instruction_override}")
            try:
                success_list, steps_list = env_runner.run(policy)
            finally:
                env_runner.close()
                wandb.finish()  # Ensure that the wandb session is closed at the end of each task

            success_rate = np.mean(success_list) * 100 if success_list else 0
            avg_steps = np.mean(steps_list) if steps_list else 0

            # Get security-related statistics from the runner (if available)
            safe_list = env_runner._last_safe_list
            safety_success_list = env_runner._last_safety_success_list
            action_trace = getattr(env_runner, '_last_action_trace', [])
            skipped_out_workspace = int(getattr(env_runner, '_last_skipped_out_workspace', 0))
            skipped_out_workspace_by_reason = getattr(env_runner, '_last_skipped_out_workspace_by_reason', {})
            safety_rate = np.mean(safe_list) * 100 if safe_list else float("nan")
            safety_success_rate = np.mean(safety_success_list) * 100 if safety_success_list else 0

            # Prepare the saved results
            validated = summarize_episodes(success_list, steps_list, safe_list, safety_success_list, args.num_episodes)
            results = {
                "task_name": task,
                "task_index": task_indices_for_eval[i] if i < len(task_indices_for_eval) else None,
                "ckpt_dir": str(ckpt_dir),
                "ckpt_epoch": args.ckpt_epoch,
                "num_episodes": args.num_episodes,
                    "seed": args.seed,
                "max_steps": args.max_steps,
                "variation": args.variation,
                "timestamp": datetime.now().isoformat(),
                "statistics": {**validated["statistics"], "skipped_out_workspace": skipped_out_workspace,
                               "skipped_out_workspace_by_reason": skipped_out_workspace_by_reason},
                "episode_details": validated["episode_details"]
            }

            # Save results to a file
            results_dir = ckpt_dir / "eval_results" / task
            results_dir.mkdir(parents=True,exist_ok=True)

            # Generate filename (including timestamp and task name)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"eval_{task}_{args.ckpt_epoch}_{timestamp_str}.json"

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"\nEvaluation completed.")
            print(f"Results saved to: {results_file}")

            # Also save the latest result file (for quick viewing)
            latest_file = results_dir / f"eval_{task}_latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Latest results saved to: {latest_file}")

            trace_payload = {
                "task_name": task,
                "task_index": task_indices_for_eval[i] if i < len(task_indices_for_eval) else None,
                "ckpt_dir": str(ckpt_dir),
                "ckpt_epoch": args.ckpt_epoch,
                "num_episodes": args.num_episodes,
                    "seed": args.seed,
                "max_steps": args.max_steps,
                "variation": args.variation,
                "timestamp": datetime.now().isoformat(),
                "action_trace": action_trace,
            }
            trace_file = results_dir / f"action_trace_{task}_{args.ckpt_epoch}_{timestamp_str}.json"
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(trace_payload, f, indent=2, ensure_ascii=False)
            latest_trace_file = results_dir / f"action_trace_{task}_latest.json"
            with open(latest_trace_file, 'w', encoding='utf-8') as f:
                json.dump(trace_payload, f, indent=2, ensure_ascii=False)
            print(f"Action traces saved to: {trace_file}")
    
    else:
        instruction_override = _get_runtime_overrides(0)
        wandb.init(
            project="vlmbench-eval",
            name=f"{task_name}_{ckpt_dir.name}_{args.ckpt_epoch}",
            config={
                "task_name": task_name,
                "ckpt_dir": str(ckpt_dir),
                "ckpt_epoch": args.ckpt_epoch,
                "num_episodes": args.num_episodes,
                    "seed": args.seed,
                "variation": args.variation,
                "task_index": task_indices_for_eval[0] if len(task_indices_for_eval) > 0 else None,
                "instruction": instruction_override,
            },
            mode="online" if args.log_wandb else "disabled"
        )

        env_runner = VLMBenchRunner(
            task_name=task_name,
            num_episodes=args.num_episodes,
            max_episode_length=args.max_steps,
            image_size=(128, 128),
            headless=args.headless,
            variation=args.variation,
            verbose=args.verbose,
            instruction=instruction_override,
        )

        # Run evaluation
        print(f"\nStarting evaluation; instruction={instruction_override}")
        try:
            success_list, steps_list = env_runner.run(policy)
        finally:
            env_runner.close()
            wandb.finish()

        success_rate = np.mean(success_list) * 100 if success_list else 0
        avg_steps = np.mean(steps_list) if steps_list else 0

        # Get security-related statistics from the runner (if available)
        safe_list = env_runner._last_safe_list
        safety_success_list = env_runner._last_safety_success_list
        action_trace = getattr(env_runner, '_last_action_trace', [])
        skipped_out_workspace = int(getattr(env_runner, '_last_skipped_out_workspace', 0))
        skipped_out_workspace_by_reason = getattr(env_runner, '_last_skipped_out_workspace_by_reason', {})
        safety_rate = np.mean(safe_list) * 100 if safe_list else float("nan")
        safety_success_rate = np.mean(safety_success_list) * 100 if safety_success_list else 0

        # Prepare the saved results
        validated = summarize_episodes(success_list, steps_list, safe_list, safety_success_list, args.num_episodes)
        results = {
            "task_name": task_name,
            "task_index": task_indices_for_eval[0] if len(task_indices_for_eval) > 0 else None,
            "ckpt_dir": str(ckpt_dir),
            "ckpt_epoch": args.ckpt_epoch,
            "num_episodes": args.num_episodes,
                    "seed": args.seed,
            "max_steps": args.max_steps,
            "variation": args.variation,
            "timestamp": datetime.now().isoformat(),
            "statistics": {**validated["statistics"], "skipped_out_workspace": skipped_out_workspace,
                           "skipped_out_workspace_by_reason": skipped_out_workspace_by_reason},
            "episode_details": validated["episode_details"]
        }

        # Save results to a file
        results_dir = ckpt_dir / "eval_results" / task_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename (including timestamp and task name)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"eval_{task_name}_{args.ckpt_epoch}_{timestamp_str}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed.")
        print(f"Results saved to: {results_file}")

        # Also save the latest result file (for quick viewing)
        latest_file = results_dir / f"eval_{task_name}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Latest results saved to: {latest_file}")

        trace_payload = {
            "task_name": task_name,
            "task_index": task_indices_for_eval[0] if len(task_indices_for_eval) > 0 else None,
            "ckpt_dir": str(ckpt_dir),
            "ckpt_epoch": args.ckpt_epoch,
            "num_episodes": args.num_episodes,
                    "seed": args.seed,
            "max_steps": args.max_steps,
            "variation": args.variation,
            "timestamp": datetime.now().isoformat(),
            "action_trace": action_trace,
        }
        trace_file = results_dir / f"action_trace_{task_name}_{args.ckpt_epoch}_{timestamp_str}.json"
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)
        latest_trace_file = results_dir / f"action_trace_{task_name}_latest.json"
        with open(latest_trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)
        print(f"Action traces saved to: {trace_file}")

if __name__ == "__main__":
    main()
