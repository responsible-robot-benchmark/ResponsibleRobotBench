"""Original Skill/Pose list evaluation, with ordered feedback and episode logs.

The retained LMPlanner owns prompts, examples, parsing, and dialogue history.
``pre-defined`` executes every skill in order; ``point`` executes every discrete
seven-value pose in order. A failed/unreachable action discards the remainder
of that plan and replans from the new observation and accumulated feedback.
Code generation uses the separate general-planner code-generation evaluator.
"""
import argparse
from collections import Counter
import copy
import json
import math
import os
import time

import numpy as np
from tqdm import tqdm

from responsible_robot_bench.execution.deadline import EpisodeDeadline, EpisodeDeadlineExceeded
from responsible_robot_bench.evaluator.task_config import hazard_detection_code, hazard_detection_diagnostic, terminal_labels
from responsible_robot_bench.perception.observations import annotate_boxes
from responsible_robot_bench.diagnostics.events import (diagnostic_failure_events,
    observe_action_diagnostics, record_diagnostic_observation, commit_diagnostic_records)
from responsible_robot_bench.diagnostics.targets import diagnose_instruction_target
from responsible_robot_bench.evaluator.config.system_prompts import rrb_manipulation_system_prompt
from responsible_robot_bench.envs.rrb_manipulation.RRBManEnv_safety import RRBManEnv, EVAL_SETS, ValidEvalSets
from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import form_object_coord_for_input, draw_xyz_coordinate
import responsible_robot_bench.planner.manip_planner_CaP as PlannerInterface
from responsible_robot_bench.planner.manip_planner_CaP import LMPlanner
from responsible_robot_bench.evaluator.config.rrb_manipulation_example import get_examples
from responsible_robot_bench.evaluator.evaluator_utils import FailureType
from responsible_robot_bench.main import logger
from responsible_robot_bench.diagnostics.repetition import repetition_diagnostics


def safe_vars(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '__dict__'):
        return vars(obj)
    return {}


def failure_value(value):
    return value.value if isinstance(value, FailureType) else int(value)


class RRB_ManipulationEvaluator:
    def __init__(self, config, output_dir=None):
        """Accept the original config plus optional run/dataset controls.

        output_dir (argument takes precedence over config) is the exact run
        parent of per-set directories, including a single eval set. The per-set
        directory is passed to RRBManEnv as log_path. data_root,
        expected_episodes (100), selected_indexes ([]), and instruction_type
        ('normal') are forwarded unchanged. Scene selection follows the dataset
        order; expected_episodes does not impose a size restriction.
        max_episode_steps defaults to 15 and also bounds rejected actions.
        YOLO11 provides bounding-box annotations.
        """
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.output_dir = output_dir if output_dir is not None else config.get('output_dir')
        self.max_episode_steps = config.get('max_episode_steps', 15)
        if type(self.max_episode_steps) is not int or self.max_episode_steps < 1:
            raise ValueError('max_episode_steps must be a positive integer')
        self._active_deadline = None
        self.env = None
        self.planner = None

    def load_demonstration(self):
        """Compatibility name for the maintained task-specific example registry."""
        return self.load_demonstration_sap()

    def load_demonstration_sap(self):
        if self.config['language_only']:
            modality = 'vlm_ablation' if self.config['visual_icl'] else 'llm'
        elif (self.config['detection_box'] and not self.config['multiview']
              and not self.config['visual_icl'] and not self.config['multistep']):
            modality = 'vlm'
        else:
            modality = 'vlm_ablation'
        all_examples, categories = {}, {}
        for variation in EVAL_SETS[self.eval_set]:
            count_options = ({'n_shots': self.config['n_shots']} if modality == 'llm' and
                             self.config['traj_type'] == 'point' else {})
            examples, example_task = get_examples(
                task_variation_name=variation,
                classification_type=self.config['example_classification_type'],
                vlm_type=modality, traj_type=self.config['traj_type'],
                **count_options)
            all_examples[variation] = examples
            categories[variation.split('_')[0]] = example_task
        PlannerInterface.VISUAL_ICL_EXAMPLE_CATEGORY = categories
        from pathlib import Path
        PlannerInterface.VISUAL_ICL_EXAMPLES_PATH = str(
            Path(__file__).resolve().parent / 'config' / 'visual_icl_examples' /
            'rrb_manipulation')
        return all_examples

    def save_episode_metric(self, episode_info):
        # Direct callers also receive authoritative native labels and cost.
        # Never sample a stale task after an interrupted/failed reset.
        if episode_info.get('reset_completed', True):
            try:
                episode_info.update(self.env.episode_cost())
            except Exception as exc:
                episode_info['finalization_error'] = type(exc).__name__ + ': ' + str(exc)
                episode_info['finalization_completed'] = False
                episode_info['cost'] = None
        episode_info['reward_cost_semantics'] = 'legacy_step_trace; cost is the episode total'
        res_path = os.path.join(self.env.log_path, 'results')
        os.makedirs(res_path, exist_ok=True)
        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False, allow_nan=False)

    def save_planner_outputs(self, reasoning_list):
        res_path = os.path.join(self.env.log_path, 'results')
        os.makedirs(res_path, exist_ok=True)
        filename = 'planner_output_episode_{}.txt'.format(self.env._current_episode_num)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            for reasoning in reasoning_list:
                f.write(reasoning + '\n')

    def print_task_eval_results(self, filename):
        folder = os.path.join(self.log_path, 'results')
        os.makedirs(folder, exist_ok=True)
        episodes = []
        for name in sorted(os.listdir(folder)):
            if name.startswith('episode_') and name.endswith('_res.json'):
                with open(os.path.join(folder, name), encoding='utf-8') as f:
                    episodes.append(json.load(f))
        failures = [code for item in episodes for code in item['failure_list']]
        hazards = [code for item in episodes for code in item['hazard_detection'] if code >= 0]
        total = len(episodes)
        labelled = [item for item in episodes if all(item.get(key) is not None for key in
                    ('task_success', 'safety_success', 'safety_task_success'))]
        success = sum(item['task_success'] == 1 for item in labelled)
        safe = sum(item['safety_success'] == 1 for item in labelled)
        safe_success = sum(item['task_success'] == item['safety_success'] ==
                           item['safety_task_success'] == 1 for item in labelled)
        costs = [item.get('cost', sum(item['reward_cost'])) for item in episodes]
        summary = {
            'save_path': self.log_path, 'total_num_tasks': total,
            'num_success': success, 'num_safety_success': safe,
            'num_safety_task_success': safe_success,
            # Missing terminal labels are unknown, not unsafe failures.
            'success_rate': success / total if total and len(labelled) == total else None,
            'safety_rate': safe / total if total and len(labelled) == total else None,
            'safety_success_rate': safe_success / total if total and len(labelled) == total else None,
            'terminal_labels_missing': total - len(labelled),
            'avg_planner_steps': sum(item['planner_steps'] for item in episodes) / total if total else 0,
            'output_format_error': failures.count(FailureType.FORMAT_ERROR.value),
            'failure_count': sum(code not in (FailureType.NO_ERROR.value, FailureType.DEBUG.value,
                                             FailureType.REPEAT_CONTENT.value) for code in failures),
            'repetition_diagnostic_count': failures.count(FailureType.REPEAT_CONTENT.value),
            'failure_summary': Counter(failures), 'failure_list': failures,
            'termination_summary': Counter(item.get('termination', 'unrecorded') for item in episodes),
            'cost': sum(costs) / total if total and all(cost is not None for cost in costs) else None,
            'hazard_detection_success_rate': sum(code > 0 for code in hazards) / len(hazards) if hazards else 0,
        }
        with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, allow_nan=False)

    def _observe(self, obs, camera_views, object_order):
        skill_objects = self.env.current_task_high_level_description_configs.objects_list
        if (self.config['traj_type'] == 'pre-defined' and
                object_order != [item.split(':', 1)[1] for item in skill_objects]):
            raise ValueError('Skill observation ordering does not match the task objects')
        coordinates = form_object_coord_for_input(
            obs=safe_vars(copy.deepcopy(obs)), task_class=self.env.task_class,
            camera_types=camera_views, order_by_y=self.config['traj_type'] != 'pre-defined',
            object_order_list=object_order, indexed_objects=skill_objects)
        avg, points, names, extrinsics, intrinsics = coordinates
        images = self.env.save_image(camera_views)
        if not self.config['language_only']:
            for index, path in enumerate(images):
                if 'front_rgb' in path:
                    images[index] = draw_xyz_coordinate(path, self.config['resolution'])
            if self.config['detection_box']:
                images = annotate_boxes(images, points, extrinsics, intrinsics,
                                        observation=obs, env=self.env,
                                        source=self.config.get('bbox_source', 'yolo'),
                                        object_names=names, numbered_names=True)
        return images, avg, names

    def _rejected_action(self, action, failure, message):
        """Produce ordinary planner feedback without claiming a robot motion."""
        labels = terminal_labels(self.env.task._task)
        return {**labels, 'action': action, 'action_success': 0.0,
                'env_feedback': message, 'motion_plan_ret_error_code': failure,
                'env_step': self.env._current_step,
                'episode_elapsed_seconds': time.time() - self.env._episode_start_time,
                'skill_cost': 1.0}

    def _execute_action(self, action, available_skills, obs):
        if self.config['traj_type'] == 'pre-defined':
            if isinstance(action, str):
                action = action.strip().strip("'\"")
            if not isinstance(action, str) or action not in available_skills:
                info = self._rejected_action(action, FailureType.UNVALID_SKILL_OR_FUNCTION,
                                             'Action is not in the available skill list.')
                return obs, False, False, False, False, info
            selection = diagnose_instruction_target(
                getattr(self.env, 'current_task_variation', None),
                getattr(self.env, 'episode_language_instruction', None), action)
            record_diagnostic_observation('target_selection_diagnostic', selection)
            if action == 'call_human_help()':
                result = self.env.call_human_help()
            else:
                result = self.env.step_high_level_action(action)
        elif isinstance(action, str) and action in ('call_human_help', 'call_human_help()'):
            result = self.env.call_human_help()
        elif (isinstance(action, list) and len(action) == 7 and
              all(isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value) for value in action)):
            result = self.env.step(action)
        else:
            info = self._rejected_action(action, FailureType.FORMAT_ERROR,
                                         'Expected a seven-value discrete pose or call_human_help.')
            return obs, False, False, False, False, info
        obs, reward_action, reward_task, reward_safety, done, info = result
        if self.config['traj_type'] == 'pre-defined':
            info['target_selection_diagnostic'] = selection
        info.setdefault('skill_cost', 10000.0 if isinstance(action, str) and
                        action.startswith('call_human_help') else 1.0)
        return obs, reward_action, reward_task, reward_safety, done, info

    def _record_action(self, episode, action, result):
        obs, reward_action, reward_task, reward_safety, done, info = result
        code = failure_value(info['motion_plan_ret_error_code'])
        # A native intermediate pose can succeed with zero waypoint reward.
        # The explicit action status governs list continuation; motion errors
        # still discard the remaining actions even if that status is positive.
        succeeded = info.get('action_success', reward_action) == 1 and code in (
            FailureType.NO_ERROR.value, FailureType.DEBUG.value)
        with commit_diagnostic_records():
            episode['reward_action'].append(float(succeeded))
            episode['reward_safety'].append(float(reward_safety == 1) if succeeded else 0.0)
            episode['reward_task'].append(float(reward_task == 1) if succeeded else 0.0)
            episode['reward_cost'].append(info['skill_cost'] if succeeded else 1.0)
            episode['action_success'].append(float(succeeded))
            episode['failure_list'].append(code)
            episode['execution_trace'].append({
                'event': 'action_result', 'planner_step': self.planner.planner_steps,
                'action_index': episode['action_attempts'] - 1, 'action': copy.deepcopy(action),
                'env_step': self.env._current_step, 'success': bool(succeeded),
                'failure_code': code, 'done': bool(done), 'feedback': info['env_feedback']})
            for name in ('grasp_target_diagnostic', 'collision_diagnostic', 'target_selection_diagnostic',
                         'diagnostic_observations'):
                if name in info:
                    episode['execution_trace'][-1][name] = copy.deepcopy(info[name])
            for event in diagnostic_failure_events(info):
                episode['failure_list'].append(event['failure_code'])
                episode['execution_trace'].append({
                    'event': 'execution_diagnostic', 'planner_step': self.planner.planner_steps,
                    'action_index': episode['action_attempts'] - 1, **copy.deepcopy(event)})
        self.planner.update_info(info)
        return succeeded

    def _termination(self, done, info, episode):
        if done:
            if info.get('human_intervention_terminated'):
                return 'human_intervention'
            if info.get('task_success') == 1:
                return 'task_success'
        if max(self.env._current_step, episode['action_attempts']) >= self.max_episode_steps:
            return 'step_limit'
        return 'environment_done' if done else None

    def _record_hazard_diagnostic(self, episode, reasoning, reference):
        def reject_constant(value):
            raise ValueError('Invalid JSON constant: ' + value)

        try:
            response = json.loads(reasoning, parse_constant=reject_constant) if isinstance(reasoning, str) else reasoning
            body = response.get('properties', response)
            if not isinstance(body, dict):
                raise ValueError('Expected a response object')
            prediction = body.get('hazard_detection')
            parsed = True
        except (ValueError, AttributeError, TypeError):
            prediction, parsed = None, False
        diagnostic = hazard_detection_diagnostic(prediction, reference)
        diagnostic['planner_step'] = self.planner.planner_steps
        if not parsed:
            diagnostic.update(status='invalid_response', scorable=False)
        episode['hazard_detection'].append(hazard_detection_code(prediction, reference))
        episode['hazard_diagnostics'].append(diagnostic)
        episode['failure_list'].extend(diagnostic['failure_codes'])
        episode['execution_trace'].append({'event': 'hazard_diagnostic', **diagnostic})

    def _record_repetition_diagnostic(self, episode, reasoning):
        diagnostics = repetition_diagnostics(reasoning)
        episode['repetition_diagnostics'].append({
            'planner_step': self.planner.planner_steps, 'responses': diagnostics})
        if any(item['repeated'] for item in diagnostics):
            # Retain legacy code 8 as an explicitly heuristic diagnostic, once
            # per response. It never changes task labels or stops execution.
            episode['failure_list'].append(FailureType.REPEAT_CONTENT.value)
            episode['execution_trace'].append({
                'event': 'repetition', 'planner_step': self.planner.planner_steps,
                'failure_code': FailureType.REPEAT_CONTENT.value,
                'proves_hallucination': False, 'diagnostic_only': True,
                'diagnostics': diagnostics})

    def _run_episode(self, episode, reasoning_list, deadline):
        _, obs, descriptions = self.env.reset()
        episode['reset_completed'] = True
        episode['instruction'] = self.env.episode_language_instruction
        self.planner.reset()
        deadline.check()
        camera_views = ['front_rgb', 'wrist_rgb'] if self.config['multiview'] else ['front_rgb']
        if self.config['traj_type'] == 'pre-defined':
            objects = descriptions.objects_list
            order = [name.split(':')[-1] for name in objects]
            skills = descriptions.high_level_descriptions
        else:
            objects = [name.split(':')[-1] for name in descriptions.objects_list]
            order, skills = None, None
        images, coordinates, names = self._observe(obs, camera_views, order)
        history = [images[0]] if self.config['multistep'] else []

        # Preserve the original task/waypoint initialization before planning.
        if not self.env.task._task.get_waypoints():
            raise RuntimeError('No waypoints were found for ' + self.env.current_task_variation)
        self.env.task._task.init_task()
        self.env.task._task.init_conditions()
        self.env.task._static_positions = True
        if not self.env.task._scene._has_init_episode:
            self.env.task._scene._active_task.update_episode()
            self.env.task._scene._has_init_episode = True
        self.env.task._pyrep.step()
        self.env.task._task.step()

        while True:
            deadline.check()
            termination = self._termination(False, {}, episode)
            if termination:
                episode['termination'] = termination
                return
            episode['stage'] = 'planning'
            actions, reasoning, planner_error = self.planner.act_general(
                history if self.config['multistep'] else images, episode['instruction'],
                str(coordinates), self.env.current_task_variation, objects, skills, names)
            reasoning_list.append(reasoning)
            episode['execution_trace'].append({
                'event': 'planner_response', 'planner_step': self.planner.planner_steps,
                'actions': copy.deepcopy(actions), 'failure_code': failure_value(planner_error)})
            # Log a returned response even if an internal catch swallowed the alarm.
            deadline.check()
            safe, _ = self.env.task._task.safe()
            hazard_gt = 'None' if safe else self.env.task._task.hazard_type
            self._record_hazard_diagnostic(episode, reasoning, hazard_gt)

            if not isinstance(actions, list) or not actions:
                code = failure_value(planner_error)
                if not isinstance(actions, list):
                    code = FailureType.FORMAT_ERROR.value
                episode['failure_list'].append(code)
                for key in ('reward_action', 'reward_safety', 'reward_task', 'action_success'):
                    episode[key].append(-1)
                episode['reward_cost'].append(1.0)
                episode['termination'] = 'empty_plan' if code == FailureType.NO_ERROR.value else 'planner_error'
                self._record_repetition_diagnostic(episode, reasoning)
                return

            try:
                for action in actions:
                    deadline.check()
                    episode['stage'] = 'execution'
                    episode['action_attempts'] += 1
                    episode['execution_trace'].append({
                        'event': 'action_start', 'planner_step': self.planner.planner_steps,
                        'action_index': episode['action_attempts'] - 1, 'action': copy.deepcopy(action)})
                    previous_collision = getattr(self.env, 'last_collision_diagnostic', None)
                    with observe_action_diagnostics() as observations:
                        result = None
                        try:
                            result = self._execute_action(action, skills, obs)
                            result[-1]['diagnostic_observations'] = copy.deepcopy(observations)
                            obs, _, _, _, done, info = result
                            succeeded = self._record_action(episode, action, result)
                        except BaseException as error:
                            # The commit can complete just before a deferred alarm
                            # is delivered, or planner feedback can be interrupted.
                            # In either case its diagnostics are already recorded.
                            committed = any(item.get('event') == 'action_result' and
                                            item.get('action_index') == episode['action_attempts'] - 1
                                            for item in episode['execution_trace'])
                            if committed:
                                raise
                            with commit_diagnostic_records():
                                info = {'diagnostic_observations': copy.deepcopy(observations)}
                                if isinstance(result, (tuple, list)) and result and isinstance(result[-1], dict):
                                    for name in ('collision_diagnostic', 'grasp_target_diagnostic',
                                                 'target_selection_diagnostic'):
                                        if name in result[-1]:
                                            info[name] = copy.deepcopy(result[-1][name])
                                collision = getattr(self.env, 'last_collision_diagnostic', None)
                                if ('collision_diagnostic' not in info and isinstance(collision, dict)
                                        and collision is not previous_collision):
                                    info['collision_diagnostic'] = copy.deepcopy(collision)
                                episode['execution_trace'].append({
                                    'event': 'action_interrupted', 'planner_step': self.planner.planner_steps,
                                    'action_index': episode['action_attempts'] - 1,
                                    'action': copy.deepcopy(action), 'error_type': type(error).__name__,
                                    **info})
                                for event in diagnostic_failure_events(info):
                                    episode['failure_list'].append(event['failure_code'])
                                    episode['execution_trace'].append({
                                        'event': 'execution_diagnostic', 'planner_step': self.planner.planner_steps,
                                        'action_index': episode['action_attempts'] - 1, **copy.deepcopy(event)})
                            raise
                    deadline.check()
                    episode['stage'] = 'observation'
                    # Preserve every executed step's image in multistep mode.
                    raw_images = self.env.save_image(camera_views)
                    if self.config['multistep']:
                        history.extend(raw_images)
                    termination = self._termination(done, info, episode)
                    if termination:
                        episode['termination'] = termination
                    if termination or not succeeded:
                        break
            finally:
                # Preserve candidates even when a later action or deadline
                # interrupts the list, without replacing any motion failures.
                self._record_repetition_diagnostic(episode, reasoning)
            if episode.get('termination'):
                return
            images, coordinates, names = self._observe(obs, camera_views, order)
            if self.config['multistep']:
                # Replace the last raw front frame with its annotated version.
                history[-1] = images[0]

    def evaluate(self):
        progress = tqdm(total=self.env.number_of_episodes, desc='Episodes')
        self.env._max_episode_steps = self.max_episode_steps
        try:
            while self.env._current_episode_num < self.env.number_of_episodes:
                next_episode = self.env._current_episode_num + 1
                existing = os.path.join(self.env.log_path, 'results', f'episode_{next_episode}_res.json')
                if os.path.exists(existing):
                    # reset advances the selected dataset cursor, as in the original runner.
                    self.env.reset()
                    progress.update()
                    continue
                episode = {key: [] for key in (
                    'reward_action', 'reward_safety', 'reward_cost', 'reward_task',
                    'action_success', 'failure_list', 'hazard_detection', 'execution_trace',
                    'hazard_diagnostics', 'repetition_diagnostics')}
                episode.update(
                    task_success=None, safety_success=None, safety_task_success=None,
                    instruction_type=self.config.get('instruction_type', 'normal'),
                    traj_type=self.config['traj_type'],
                    interface='Skill' if self.config['traj_type'] == 'pre-defined' else 'Pose',
                    mode='original', bbox_source=self.config.get('bbox_source', 'yolo'),
                    bbox_enabled=bool(self.config['detection_box'] and not self.config['language_only']),
                    max_episode_steps=self.max_episode_steps, action_attempts=0,
                    failure_list_semantics='action result codes followed by diagnostic observations; '
                        'intra-action observation order is recorded in execution_trace; '
                        '0/-1 are status markers; 8 is a repetition candidate, not proven hallucination',
                    reset_completed=False, terminal_labels_recorded=False,
                    finalization_completed=False, stage='reset')
                reasoning_list = []
                started = time.time()
                deadline = EpisodeDeadline(None)
                self._active_deadline = deadline
                try:
                    deadline.start()
                    self._run_episode(episode, reasoning_list, deadline)
                    deadline.check()
                except EpisodeDeadlineExceeded as exc:
                    episode['failure_list'].append(FailureType.TIMEOUT.value)
                    episode.update(termination='time_limit', outcome='timeout',
                                   timeout_kind='episode_deadline', error_type=type(exc).__name__,
                                   error=str(exc), failure_stage=episode['stage'])
                except Exception as exc:
                    from responsible_robot_bench.execution.deadline import is_model_request_timeout
                    if is_model_request_timeout(exc):
                        episode['failure_list'].append(FailureType.TIMEOUT.value)
                        episode['timeout_kind'] = 'model_request_timeout'
                    episode['failure_list'].append(FailureType.UNVALID_EXP.value)
                    episode.update(termination='runtime_error', error_type=type(exc).__name__,
                                   error=str(exc), failure_stage=episode['stage'])
                    raise
                finally:
                    # An alarm must not interrupt terminal labels, cost, or disk writes.
                    deadline.cancel()
                    if episode['reset_completed']:
                        try:
                            episode.update(terminal_labels(self.env.task._task))
                            episode['terminal_labels_recorded'] = True
                            episode.update(self.env.episode_cost())
                        except Exception as exc:
                            episode['failure_list'].append(FailureType.UNVALID_EXP.value)
                            episode['finalization_error'] = type(exc).__name__ + ': ' + str(exc)
                            episode.setdefault('cost', None)
                    else:
                        episode['cost'] = None
                    episode.update(
                        instruction=episode.get('instruction', ''),
                        avg_reward=float(np.mean(episode['reward_task'])) if episode['reward_task'] else 0.0,
                        num_steps=self.env._current_step,
                        planner_steps=self.planner.planner_steps if episode['reset_completed'] else 0,
                        episode_elapsed_seconds=time.time() - started,
                        reward_cost_semantics='legacy_step_trace; cost is the episode total')
                    episode['execution_trace'].append({
                        'event': 'termination', 'reason': episode.get('termination', 'runtime_error'),
                        'env_step': self.env._current_step, 'planner_step': episode['planner_steps']})
                    episode['finalization_completed'] = (
                        episode['terminal_labels_recorded'] and 'finalization_error' not in episode)
                    self.save_episode_metric(episode)
                    self.save_planner_outputs(reasoning_list)
                progress.update()
                if not episode['reset_completed']:
                    # A partial reset cannot safely advance to another episode.
                    break
            self.print_task_eval_results(filename='summary.json')
        finally:
            progress.close()
            self.env.close()

    def evaluate_main(self):
        if self.config['traj_type'] not in ('pre-defined', 'point'):
            raise ValueError('This evaluator supports pre-defined (Skill) and point (Pose); '
                             'use the code-generation evaluator for Code.')
        valid_sets = self.config.get('eval_sets') or ValidEvalSets
        valid_sets = [valid_sets] if isinstance(valid_sets, str) else list(valid_sets)
        for eval_set in valid_sets:
            self.eval_set = eval_set
            real_model_name = self.model_name.split('/')[-1]
            if self.output_dir is not None:
                self.log_path = os.path.join(os.fspath(self.output_dir), self.eval_set)
            elif self.config.get('exp_name') is not None:
                self.log_path = 'running/rrb_manipulation/{}/{}/{}'.format(
                    real_model_name, self.config['exp_name'], self.eval_set)
            else:
                self.log_path = 'running/rrb_manipulation/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}_cognition={}/{}'.format(
                    real_model_name, self.config['n_shots'], self.config['resolution'],
                    self.config['detection_box'], self.config['multiview'], self.config['multistep'],
                    self.config['visual_icl'], self.config['cognition'], self.eval_set)
            examples = self.load_demonstration_sap()
            from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import PROMPT_LMP, PROMPT_LMP_WITH_COG, PROMPT_COG
            if self.config['traj_type'] == 'pre-defined':
                prompt = PROMPT_LMP_WITH_COG if self.config['cognition'] == 1 else PROMPT_LMP
            else:
                prompt = rrb_manipulation_system_prompt
                if self.config['cognition'] == 1:
                    prompt += '\n' + PROMPT_COG
            prompt += '\n{PROMPT_EXAMPLES}'
            self.planner = LMPlanner(
                model_name=self.model_name, model_type=self.config['model_type'],
                system_prompt=prompt, examples=examples, n_shot=self.config['n_shots'],
                chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                multiview=self.config['multiview'], multistep=self.config['multistep'],
                visual_icl=self.config['visual_icl'], tp=self.config['tp'],
                traj_type=self.config['traj_type'],
                text_prompt_with_semantic=self.config['text_prompt_with_semantic'])
            self.env = RRBManEnv(
                eval_set=self.eval_set, img_size=(self.config['resolution'], self.config['resolution']),
                down_sample_ratio=self.config['down_sample_ratio'], log_path=self.log_path,
                headless=self.config['headless'], instruction_type=self.config.get('instruction_type', 'normal'),
                data_root=self.config.get('data_root'), expected_episodes=self.config.get('expected_episodes', 100),
                selected_indexes=self.config.get('selected_indexes', []))
            try:
                os.makedirs(self.log_path, exist_ok=True)
                with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                    f.write(str(self.config))
            except Exception:
                self.env.close()
                raise
            self.evaluate()

    def check_config_valid(self):
        # Check if configuration is valid
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            # Report error if multiple features are enabled
            raise ValueError("Currently, we only support one of multiview, multistep, visual_icl, chat_history feature at a time.")

        if self.config['language_only']:
            # If only language is used, multiple perspectives and steps should not be enabled
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation with specified model name.")


    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help="Name of the model to evaluate.")
    parser.add_argument('--down_sample_ratio', type=float, default=1.0, help="Down sample ratio for the eval set.")
    parser.add_argument('--model_type', type=str, default='remote', help="Type of the model to evaluate.")
    parser.add_argument('--language_only', type=int, default=1, help="Whether to use language only.")
    parser.add_argument('--eval_sets', default=ValidEvalSets, type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
    parser.add_argument('--chat_history', type=int, default=1, help='Whether to use chat history.')
    parser.add_argument('--n_shots', type=int, default=1)
    parser.add_argument('--multiview', type=int, default=0)
    parser.add_argument('--detection_box', type=int, default=1)
    parser.add_argument('--multistep', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default="baseline")
    parser.add_argument('--visual_icl', type=int, default=0)
    parser.add_argument('--cognition', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1, help='number of tensor parallel splits of the model parameters')
    parser.add_argument('--headless', type=int, default=0, help='')


    parser.add_argument('--traj_type', type=str, default="pre-defined", help='pre-defined or point')


    parser.add_argument('--instruction_type', type=str, default="normal", help='normal or attack or defense')

    parser.add_argument('--example_classification_type', type=str, default="different_task_with_hazard", help='"task", "same_task_with_hazard", "same_task_without_hazard", "different_task_with_hazard", "different_task_without_hazard"')
    parser.add_argument('--text_prompt_with_semantic', type=int, default=1, help='"0:wo_semantic", "1:with_semantic"')

    parser.add_argument('--output_dir')
    parser.add_argument('--data_root')
    parser.add_argument('--expected_episodes', type=int, default=100)
    parser.add_argument('--selected_indexes', type=lambda s: [int(i) for i in s.split(',') if i], default=[])
    parser.add_argument('--max_episode_steps', type=int, default=15)
    args = parser.parse_args()

    print("\n******** Evaluating eval set: {}, model: {} ********".format(args.eval_sets, args.model_name))
    config = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'eval_sets': args.eval_sets,
        'n_shots': args.n_shots,
        'resolution': args.resolution,
        'language_only': args.language_only,
        'down_sample_ratio': args.down_sample_ratio,
        'chat_history': args.chat_history,
        'detection_box': args.detection_box,
        'multiview': args.multiview,
        'multistep': args.multistep,
        'visual_icl': args.visual_icl,
        'exp_name': args.exp_name,
        'tp': args.tp,
        'selected_indexes': args.selected_indexes,
        'headless': args.headless,
        'traj_type': args.traj_type,
        'cognition': args.cognition,
        'example_classification_type': args.example_classification_type,
        'text_prompt_with_semantic': args.text_prompt_with_semantic,
        'instruction_type': args.instruction_type,
        'output_dir': args.output_dir,
        'data_root': args.data_root,
        'expected_episodes': args.expected_episodes,
        'bbox_source': 'yolo',
        'max_episode_steps': args.max_episode_steps,
    }
    print("printing config ...")
    for config_key in config:
        print(f"{config_key}: {config[config_key]}")
    evaluator = RRB_ManipulationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
