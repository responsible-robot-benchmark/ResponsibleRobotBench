'VLMBench environment runner - used to evaluate trained policies in the amsolver simulation environment'
import sys
from pathlib import Path

# Add necessary paths
_project_root = Path(__file__).resolve().parents[4]
_rrb_manipulation = _project_root / 'responsible_robot_bench' / 'envs' / 'rrb_manipulation'
for _path in [str(_project_root), str(_rrb_manipulation)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import wandb

from amsolver import ObservationConfig
from amsolver.action_modes import ActionMode, ArmActionMode
from amsolver.environment import Environment
from amsolver.backend.utils import task_file_to_task_class
from pyrep.const import RenderMode

from pfp.policy.base_policy import BasePolicy
from pfp.common.se3_utils import rot6d_to_quat_np
import json
from responsible_robot_bench.evaluator.task_config import terminal_labels


class VLMBenchRunner:
    'VLMBench environment runner - executes policies and evaluates in the amsolver simulation environment'
    
    def __init__(
        self,
        task_name: str,
        num_episodes: int = 10,
        max_episode_length: int = 100,
        image_size: tuple = (128, 128),
        headless: bool = True,
        variation: int = 0,
        verbose: bool = False,
        instruction: str = None,
    ):
        "Args:\n    task_name: Task name, e.g., 'charge2_inseration'\n    num_episodes: Number of episodes to evaluate\n    max_episode_length: Maximum number of steps per episode\n    image_size: Image size\n    headless: Whether to run in headless mode\n    variation: Task variant\n    verbose: Whether to print detailed information\n    instruction_file: Path to instruction file containing task instructions"
        if num_episodes <= 0 or max_episode_length <= 0:
            raise ValueError('Episode count and maximum episode length must be positive')
        self.task_name = task_name
        self.requested_task_name = task_name
        self.env_task_name = task_name
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.image_size = image_size
        self.headless = headless
        self.variation = variation
        self.verbose = verbose
        self.instructions = []
        if instruction is not None:
            # Directly use the input string
            self.instructions = [instruction]
            print(f"Instruction override: {instruction}")
        self.last_obs = None  # Save the last observation for IK error handling
        self.original_cwd = None  # Save the original working directory
        self.last_step_error = None  # Save the reason for the last step failure
        
        # Initialize the environment
        self._init_env()
        
    def _init_env(self):
        'Initialize the amsolver environment'
        # Switch to the rrb_manipulation directory (amsolver needs to run in this directory)
        import os
        self.original_cwd = os.getcwd()
        rrb_manipulation_dir = _rrb_manipulation
        os.chdir(str(rrb_manipulation_dir))
        
        # Configure the observation
        obs_config = ObservationConfig()
        obs_config.set_all(False)
        obs_config.gripper_pose = obs_config.gripper_matrix = obs_config.gripper_open = True
        obs_config.joint_positions = True
        for camera_name in ['front','left_shoulder','right_shoulder','overhead','wrist']:
            getattr(obs_config,camera_name+'_camera').rgb = True
        obs_config.right_shoulder_camera.image_size = list(self.image_size)
        obs_config.left_shoulder_camera.image_size = list(self.image_size)
        obs_config.overhead_camera.image_size = list(self.image_size)
        obs_config.wrist_camera.image_size = list(self.image_size)
        obs_config.front_camera.image_size = list(self.image_size)
        
        obs_config.right_shoulder_camera.depth_in_meters = False
        obs_config.left_shoulder_camera.depth_in_meters = False
        obs_config.overhead_camera.depth_in_meters = False
        obs_config.wrist_camera.depth_in_meters = False
        obs_config.front_camera.depth_in_meters = False
        
        # Rendering mode
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
        
        # Create the environment with end-effector pose control and path planning.
        self.env = Environment(
            action_mode=ActionMode(arm=ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME),
            obs_config=obs_config,
            headless=self.headless
        )
        self.env.launch()
        
        # Load the requested benchmark task.
        task_class = task_file_to_task_class(self.env_task_name, parent_folder='vlm')
        self.task = self.env.get_task_data_collection(task_class)
        self.task.set_variation(self.variation)
        
        print(
            f"Environment initialized: requested={self.requested_task_name}, "
            f"loaded={self.env_task_name}, variation {self.variation}"
        )

    def step(self, robot_state: np.ndarray) -> tuple:
        "Execute actions (completely following rlbench_env.py's way).\n\nArgs:\n    robot_state: Robot state (10,) = [x, y, z, r00, r10, r20, r01, r11, r21, gripper]\n\nReturns:\n    (reward, terminate): Reward and whether terminated"
        self.last_step_error = None
        self.last_executed_state = None
        errors = []

        def _build_action(state10: np.ndarray) -> np.ndarray:
            ee_position = state10[:3]
            ee_quat = rot6d_to_quat_np(state10[3:9])  # rot6d -> quat
            # gripper for simulator should be close to binary control
            gripper = np.array([1.0 if float(state10[9]) >= 0.5 else 0.0], dtype=np.float32)
            return np.concatenate([ee_position, ee_quat, gripper])

        def _try_state(state10: np.ndarray) -> tuple[bool, tuple]:
            action8 = _build_action(state10)
            try:
                obs, reward, terminate = self.task.step(action8)
                self.last_obs = obs
                self.last_executed_state = state10.astype(np.float32).tolist()
                return True, (reward, terminate)
            except Exception as e:
                if type(e).__name__ not in {'DemoError', 'ConfigurationError', 'ConfigurationPathError', 'IKError', 'InvalidActionError'}:
                    raise
                errors.append(f"{type(e).__name__}: {str(e)}")
                return False, (0.0, True)

        # 1) first try raw predicted state
        ok, ret = _try_state(robot_state)
        if ok:
            return ret

        # 2) fallback: shorten motion and make orientation conservative
        if self.last_obs is not None:
            cur_m = self.last_obs.gripper_matrix
            cur_pos = cur_m[:3, 3].astype(np.float32)
            cur_rot6d = cur_m[:3, :2].flatten(order='F').astype(np.float32)
            cur_grip = float(self.last_obs.gripper_open)

            tgt_pos = robot_state[:3].astype(np.float32)
            tgt_rot6d = robot_state[3:9].astype(np.float32)
            tgt_grip = float(robot_state[9])

            # cap large jump to improve planner feasibility
            max_delta = 0.04
            d = tgt_pos - cur_pos
            n = float(np.linalg.norm(d))
            if n > max_delta and n > 1e-8:
                tgt_pos = cur_pos + d / n * max_delta

            alphas = [0.75, 0.5, 0.25]
            for a in alphas:
                s_blend = np.concatenate([
                    cur_pos + a * (tgt_pos - cur_pos),
                    cur_rot6d + a * (tgt_rot6d - cur_rot6d),
                    np.array([tgt_grip], dtype=np.float32),
                ])
                ok, ret = _try_state(s_blend)
                if ok:
                    return ret

                s_keep_ori = np.concatenate([
                    cur_pos + a * (tgt_pos - cur_pos),
                    cur_rot6d,
                    np.array([cur_grip], dtype=np.float32),
                ])
                ok, ret = _try_state(s_keep_ori)
                if ok:
                    return ret

        # All retries failed -> terminate this episode
        self.last_step_error = " | ".join(errors[-3:]) if errors else "Unknown step error"
        if self.verbose:
            print(f"\nAction execution failed: {self.last_step_error}")
        return 0.0, True
    
    
    def _get_observation(self, obs):
        'Extract data required for the strategy from the amsolver observation'
        # Extract images from 5 cameras
        camera_names = ['front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist']
        images = []
        for cam_name in camera_names:
            img = getattr(obs, f'{cam_name}_rgb')
            # Resize
            img = Image.fromarray(img)
            if img.size != self.image_size:
                img = img.resize(self.image_size, Image.BILINEAR)
            img_array = np.array(img)
            # Convert to float32 [0, 1] to match the zarr data format used during training
            if img_array.dtype == np.uint8:
                img_array = img_array.astype(np.float32) / 255.0
            elif img_array.max() > 1.0:
                img_array = img_array.astype(np.float32) / 255.0
            else:
                img_array = img_array.astype(np.float32)
            images.append(img_array)
        
        images = np.stack(images)  # (5, H, W, 3), float32 [0, 1]
        
        # Extract robot state (xyz + rot6d + gripper)
        # Maintain consistency with the vlmbench_dataset.py and original RLBench logic used during training
        gripper_matrix = obs.gripper_matrix
        ee_position = gripper_matrix[:3, 3]  # xyz
        ee_rot6d = gripper_matrix[:3, :2].flatten(order='F')  # Flatten in column order, consistent with training
        gripper = np.array([obs.gripper_open])
        
        # Ensure dtype is float32, consistent with training
        robot_state = np.concatenate([ee_position, ee_rot6d, gripper]).astype(np.float32)  # (10,)
        
        return images, robot_state

    def run(self, policy: BasePolicy):
        'Run evaluation'
        if wandb.run is not None:
            for metric in ['success rate','average steps','safety rate','safety success rate']:
                wandb.define_metric(metric,summary='mean')
        
        success_list = []
        steps_list = []
        safe_list = []
        safety_success_list = []
        action_trace = []
        
        # Store in instance variable for external access
        self._last_success_list = success_list
        self._last_steps_list = steps_list
        self._last_safe_list = safe_list
        self._last_safety_success_list = safety_success_list
        self._last_action_trace = action_trace
        
        
        skipped_out_workspace = 0
        skipped_out_workspace_by_reason = {}
        attempt = 0
        pbar = tqdm(total=self.num_episodes, desc=f"Evaluating {self.task_name}")
        for episode in range(self.num_episodes):
            attempt += 1
            # Reset environment and policy
            descriptions, obs = self.task.reset()
            self.last_obs = obs  # Save initial observation
            policy.reset_obs()
            
            if self.instructions:
                instruction = np.random.choice(self.instructions)
                print(f"\n[Episode {episode}] Instruction: {instruction}")
            else:
                instruction = descriptions[0]

            success = False
            skip_episode = False
            skip_info = None
            episode_trace = []

            for step in range(self.max_episode_length):
                # Obtain observation (no need to add batch dimension, policy.predict_action will automatically handle it)
                images, robot_state = self._get_observation(obs)
                with torch.no_grad():
                    prediction = policy.predict_action(images, robot_state)
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.detach().cpu().numpy()
                prediction = np.asarray(prediction)
                if (prediction.ndim != 3 or prediction.shape[-1] != 10
                        or prediction.shape[1] < 2 or not np.isfinite(prediction).all()):
                    raise ValueError("Image FM must predict a finite (K, T>=2, 10) trajectory")
                prediction_np = prediction[-1]
                next_robot_state = prediction_np[1]
                
                # Print debug information (always print the first step, then every 5 steps)
                if step == 0 or (self.verbose and step % 5 == 0):
                    current_pos = obs.gripper_matrix[:3, 3]
                    target_pos = next_robot_state[:3]
                    print(f"\n  Step {step}:")
                    print(f"    Current pos: {current_pos}")
                    print(f"    Target pos:  {target_pos}")
                    print(f"    Distance:    {np.linalg.norm(target_pos - current_pos):.4f}")
                    print(f"    Gripper:     {next_robot_state[9]:.2f}")
                    
                    # Print the first few steps of the predicted trajectory (if available)
                    if prediction_np.shape[0] > 1:
                        n_show = min(5, prediction_np.shape[0])
                        print(f"    Predicted trajectory (first {n_show} of {prediction_np.shape[0]} steps):")
                        for t in range(n_show):
                            pos = prediction_np[t, :3]
                            grip = prediction_np[t, 9]
                            print(f"      t={t}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], grip={grip:.2f}")
                
                # Execute the action (completely according to rlbench_runner.py)
                reward, terminate = self.step(next_robot_state)
                # If step fails, last_obs may be empty, keep the current obs to avoid subsequent crashes
                if self.last_obs is not None:
                    obs = self.last_obs
                if terminate and self.last_step_error is not None:
                    skip_episode = True
                    skip_info = {
                        "reason": "controller_no_path_or_stuck",
                        "detail": str(self.last_step_error),
                    }
                    print(f"[SKIP] Episode {episode}: reason=controller_no_path_or_stuck, detail={self.last_step_error}")
                    break

                # Checker failures invalidate the evaluation; they are never Safe labels.
                safe, safe_term = self.task._task.safe()

                # Only record actions steps that were actually executed, not the predicted trajectory points that were not executed.
                episode_trace.append({
                    "episode": int(episode),
                    "step": int(step),
                    "predicted_robot_state": next_robot_state.astype(np.float32).tolist(),
                    "executed_robot_state": self.last_executed_state,
                    "observed_gripper_pose": np.asarray(obs.gripper_pose).tolist(),
                    "reward": float(reward),
                    "terminate": bool(terminate),
                    "safe": bool(safe),
                })
                
                if reward == 1.0:
                    success = True
                    if self.verbose:
                        print(f"Episode {episode}: success; steps: {step + 1}")
                    break
                
                # If the safety mechanism requires termination (e.g., due to triggering a dangerous condition), then the episode ends with failure.
                if safe_term:
                    if self.verbose:
                        print(f"Episode {episode}: safety termination (steps: {step + 1})")
                    terminate = True
                
                if terminate:
                    if self.verbose:
                        print(f"Episode {episode}: failure (steps: {step + 1})")
                    break
            
            if skip_episode:
                skipped_out_workspace += 1
                if skip_info is not None:
                    reason = str(skip_info.get("reason", "unknown"))
                    skipped_out_workspace_by_reason[reason] = skipped_out_workspace_by_reason.get(reason, 0) + 1
                print(f"[TERMINATED] Episode {episode}: counted, reasons={skipped_out_workspace_by_reason}")

            if not success and self.verbose:
                print(f"Episode {episode}: step limit reached")
            
            # Count every attempted episode using its actual terminal checkers.
            labels = terminal_labels(self.task._task)
            success = bool(labels['task_success'])
            safe_final = bool(labels['safety_success'])
            steps_list.append(step + 1)
            success_list.append(bool(success))
            safe_list.append(bool(safe_final))
            # safety_success = success AND safety (must be satisfied simultaneously)
            # Note: If safety_success is set in the loop (success or safe termination), this will overwrite.
            # But the final judgment should be based on success and safe_final.
            safety_success = bool(success) and bool(safe_final)
            safety_success_list.append(bool(safety_success))
            action_trace.append({
                "episode": int(episode),
                "instruction": instruction,
                "task_name": self.task_name,
                "loaded_task_name": self.env_task_name,
                "success": bool(success),
                "safe": bool(safe_final),
                "safety_success": bool(safety_success),
                "early_termination": skip_info if skip_episode else None,
                "executed_steps": len(episode_trace),
                "attempted_steps": step + 1,
                "steps": episode_trace,
            })
            episode_metrics = {
                "episode": episode,
                "success rate": int(bool(success)),
                "average steps": step + 1,
                "safety rate": int(bool(safe_final)),
                "safety success rate": int(bool(safety_success))
            }
            if wandb.run is not None:
                wandb.log(episode_metrics)
            pbar.update(1)
        pbar.close()

        # Print statistics
        success_rate = np.mean(success_list) * 100 if success_list else 0
        safety_rate = np.mean(safe_list) * 100 if safe_list else 0
        avg_steps = np.mean(steps_list) if steps_list else 0
        safety_success_rate = np.mean(safety_success_list) * 100 if safety_success_list else 0
        
        print(f"\n=== Evaluation results ===")
        print(f"Task: {self.task_name}")
        print(f"Success rate: {success_rate:.1f}% ({int(np.sum(success_list))}/{len(success_list)})")
        print(f"Safety rate: {safety_rate:.1f}% ({int(np.sum(safe_list))}/{len(safe_list)})")
        if steps_list:
            print(f"Mean steps: {avg_steps:.1f}")
        print(f"Safe success rate: {safety_success_rate:.1f}% ({int(np.sum(safety_success_list))}/{len(safety_success_list)})")
        if skipped_out_workspace > 0:
            print(f"Early terminations: {skipped_out_workspace}; all included in {self.num_episodes} episodes")
            print(f"Early termination reasons: {skipped_out_workspace_by_reason}")
        
        # Ensure instance variables are updated (though they reference the same list, for clarity, they are reassigned here)
        self._last_success_list = success_list
        self._last_steps_list = steps_list
        self._last_safe_list = safe_list
        self._last_safety_success_list = safety_success_list
        self._last_action_trace = action_trace
        self._last_terminated_early = skipped_out_workspace
        self._last_terminated_early_by_reason = skipped_out_workspace_by_reason
        # Compatibility counters remain zero because all attempted episodes are reported.
        self._last_skipped_out_workspace = 0
        self._last_skipped_out_workspace_by_reason = {}

        return success_list, steps_list
    
    def close(self):
        'Close environment'
        if hasattr(self, 'env') and self.env is not None:
            self.env.shutdown()
        # Restore original working directory
        if self.original_cwd is not None:
            import os
            os.chdir(self.original_cwd)
