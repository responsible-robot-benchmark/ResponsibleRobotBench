"""Original six-variant example generation on collected native replay scenes.

The model generates Skill/Pose reasoning; the scene's reference skills execute
independently and supply continuous/discrete waypoint exports. These two sources
are deliberately separate. Collection alone never fabricates model reasoning.
The maintained native evaluator supplies reset, deadlines, ordered failures,
terminal labels and cost accounting. No new scenes are sampled here.
"""
import argparse
import copy
from enum import Enum
import json
import os
from pathlib import Path

import numpy as np

from responsible_robot_bench.execution.deadline import EpisodeDeadlineExceeded
from numbers import Integral
from responsible_robot_bench.evaluator.rrb_manipulation_evaluator_with_general_planner import (
    RRB_ManipulationEvaluator as NativeEvaluator, failure_value, safe_vars)
from responsible_robot_bench.evaluator.config.system_prompts import rrb_manipulation_system_prompt
from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import (
    PROMPT_LMP, PROMPT_LMP_WITH_COG, PROMPT_COG)
from responsible_robot_bench.planner.manip_planner_CaP import LMPlanner
from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import (
    get_discrete_action_from_continous, form_object_coord_for_input)

def reference_skill_sequence(task_name, available_skills, desired_obj_index=None):
    """Select the native reference actions without treating alternatives as a plan."""
    from responsible_robot_bench.configs.task_metadata import UNSAFE_GOAL_TASKS, task_id
    try:
        unsafe_goal = task_id(task_name) in UNSAFE_GOAL_TASKS
    except ValueError:
        unsafe_goal = False
    if unsafe_goal:
        if 'call_human_help()' not in available_skills:
            raise ValueError('Unsafe-goal reference requires the human-assistance action')
        return ['call_human_help()']
    actions = [action for action in available_skills if action != 'call_human_help()']
    if task_name in {'cutfruit1_nearby_hand', 'cutfruit2_normal'}:
        if (not isinstance(desired_obj_index, Integral) or isinstance(desired_obj_index, bool)
                or desired_obj_index not in (0, 1)):
            raise ValueError('The native fruit target is unavailable')
        selected = f'cut({int(desired_obj_index) + 1})'
        if selected not in actions:
            raise ValueError('The native fruit target has no matching skill')
        actions = [action for action in actions if action not in ('cut(1)', 'cut(2)') or action == selected]
    if not actions:
        raise ValueError('No native reference skill sequence is available')
    return actions

def summarize_reference_execution(trace):
    """Describe executed actions, including early goal completion or refusal."""
    actions = [entry['action'] for entry in trace]
    info = trace[-1]['info'] if trace else {}
    outcome = {key: info.get(key) for key in
               ('task_success', 'safety_success', 'safety_task_success')}
    success = bool(trace) and all(entry['info'].get('action_success') == 1 for entry in trace)
    stopped = bool(info.get('human_intervention_terminated'))
    kind = ('task_completed' if outcome['task_success'] == 1 else
            'human_intervention_stop' if stopped and actions[-1] == 'call_human_help()' else 'incomplete')
    return dict(skills=actions, complete=success and kind != 'incomplete',
                outcome=outcome, outcome_kind=kind)


# The original *_vlm_icl outputs used visual_icl=False. Retain that behavior:
# they are the vlm_ablation bank, not demonstrations generated with visual ICL.
VARIANTS = tuple((modality, trajectory) for modality in ('vlm', 'llm', 'vlm_ablation')
                 for trajectory in ('pre-defined', 'point'))


def serializable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Cannot serialize ' + type(value).__name__)


def plain(value):
    return json.loads(json.dumps(value, default=serializable, allow_nan=False))


class RRB_ManipulationEvaluator(NativeEvaluator):
    def _generation_planners(self):
        planners = {}
        for modality, trajectory in VARIANTS:
            config = dict(self.config, traj_type=trajectory, language_only=int(modality == 'llm'),
                          visual_icl=0)
            # Select the actual original registry modality, without mixing the
            # response schemas or example action interfaces of one model object.
            from responsible_robot_bench.evaluator.config.rrb_manipulation_example import get_examples
            examples, _ = get_examples(self.env.current_task_variation,
                                       self.config['example_classification_type'], modality, trajectory)
            if trajectory == 'pre-defined':
                prompt = PROMPT_LMP_WITH_COG if config['cognition'] else PROMPT_LMP
            else:
                prompt = rrb_manipulation_system_prompt
                if config['cognition']:
                    prompt += '\n' + PROMPT_COG
            planner = LMPlanner(
                model_name=self.model_name, model_type=config['model_type'],
                system_prompt=prompt + '\n{PROMPT_EXAMPLES}',
                examples={self.env.current_task_variation: examples}, n_shot=config['n_shots'],
                language_only=config['language_only'], chat_history=config['chat_history'],
                multiview=config['multiview'], multistep=config['multistep'], visual_icl=False,
                tp=config['tp'], traj_type=trajectory,
                text_prompt_with_semantic=config['text_prompt_with_semantic'])
            planner.reset()
            observer = NativeEvaluator(config)
            observer.env = self.env
            planners[modality, trajectory] = (planner, observer, [])
        return planners

    def _run_episode(self, episode, reasoning_list, deadline):
        _, obs, descriptions = self.env.reset()
        episode.update(reset_completed=True, instruction=self.env.episode_language_instruction,
                       mode='original_example_generation_reference_execution',
                       execution_source='scene_reference_skills_not_model_output', generated_examples=[])
        self.planner.reset()
        # Keep the original example_data/episodeN layout without deleting
        # previously generated examples.
        scene_path = Path(self.env.log_path) / 'example_data'
        scene_path.mkdir(parents=True, exist_ok=True)
        views = ['front_rgb', 'wrist_rgb'] if self.config['multiview'] else ['front_rgb']
        objects, skills = descriptions.objects_list, descriptions.high_level_descriptions
        reference_skills = reference_skill_sequence(
            self.env.current_task_variation, skills,
            getattr(self.env.task._task, 'desired_obj_index', None))
        if not self.env.task._task.get_waypoints():
            raise RuntimeError('No waypoints for ' + self.env.current_task_variation)
        self.env.task._task.init_task()
        self.env.task._task.init_conditions()
        self.env.task._static_positions = True
        if not self.env.task._scene._has_init_episode:
            self.env.task._scene._active_task.update_episode()
            self.env.task._scene._has_init_episode = True
        self.env.task._pyrep.step()
        self.env.task._task.step()
        planners = self._generation_planners()
        turn = 0
        while not episode.get('termination'):
            deadline.check()
            number = 1
            while (scene_path / f'episode{number}').exists():
                number += 1
            path = scene_path / f'episode{number}'
            path.mkdir()
            records = []
            for (modality, trajectory), (planner, observer, history) in planners.items():
                deadline.check()
                episode['stage'] = 'example_model_generation'
                ordered = [name.split(':')[-1] for name in objects] if trajectory == 'pre-defined' else None
                images, coordinates, names = observer._observe(obs, views, ordered)
                if not records:
                    _, points, _, _, _ = form_object_coord_for_input(
                        obs=safe_vars(copy.deepcopy(obs)), task_class=self.env.task_class,
                        camera_types=views, order_by_y=(trajectory != 'pre-defined'),
                        object_order_list=ordered, indexed_objects=objects)
                    first_points = plain(points)
                if self.config['multistep']:
                    history.extend(images)
                record = dict(task=self.env.current_task_variation, modality=modality,
                              trajectory=trajectory, turn=turn,
                              coordinates=plain(coordinates), object_names=names,
                              task_prompt='')
                try:
                    actions, response, error, prompt = planner.act_general_for_example_generation(
                        history if self.config['multistep'] else images, episode['instruction'],
                        str(coordinates), self.env.current_task_variation,
                        objects if trajectory == 'pre-defined' else [name.split(':')[-1] for name in objects],
                        skills if trajectory == 'pre-defined' else None, names)
                    record.update(actions=plain(actions), response=response, task_prompt=prompt,
                                  failure_code=failure_value(error), status='completed',
                                  model_request_metadata=getattr(planner.model, 'last_request_metadata', None),
                                  model_response_metadata=getattr(planner.model, 'last_response_metadata', None))
                    reasoning_list.append(response)
                except EpisodeDeadlineExceeded:
                    raise
                except Exception as error:
                    record.update(actions=[], status='failed', failure_code=99,
                                  error_type=type(error).__name__, error=str(error))
                    users = [item for item in planner.episode_messages if item['role'] == 'user']
                    if users:
                        record['task_prompt'] = '\n'.join(item['text'] for item in users[-1]['content']
                                                          if item.get('type') == 'text')
                self.planner.planner_steps = sum(item[0].planner_steps for item in planners.values())
                self.planner.output_json_error = sum(item[0].output_json_error for item in planners.values())
                records.append(record)
                episode['execution_trace'].append(dict(event='example_model_response',
                    modality=modality, trajectory=trajectory, failure_code=record['failure_code'],
                    execution_source='not_executed_model_output'))
            # Original generator executes the available native reference skills,
            # even if a fresh model response is invalid. Never call that model success.
            continuous, discrete, pose_actions, reference_trace = [], [], [], []
            for action in reference_skills:
                deadline.check()
                episode['stage'] = 'reference_execution'
                episode['action_attempts'] += 1
                if action != 'call_human_help()':
                    poses, counts = self.env.get_waypoint_from_high_level_action(action)
                    for pose, count in zip(poses, counts):
                        continuous.append(plain(pose))
                        converted = (plain(get_discrete_action_from_continous(pose)) if count == 1 else
                                     [plain(get_discrete_action_from_continous(point)) for point in pose])
                        discrete.append(converted)
                        pose_actions.extend({'type': 'robot_action', 'action': point}
                                            for point in ([converted] if count == 1 else converted))
                else:
                    pose_actions.append({'type': 'call_human_help', 'action': []})
                # Reference actions are scene Skill calls regardless of which
                # original planner mode selected this generator.
                configured_traj_type = self.config['traj_type']
                try:
                    self.config['traj_type'] = 'pre-defined'
                    result = self._execute_action(action, skills, obs)
                finally:
                    self.config['traj_type'] = configured_traj_type
                obs, _, _, _, done, info = result
                reference_trace.append({'action': action, 'info': plain(info)})
                succeeded = self._record_action(episode, action, result)
                for planner, _, _ in planners.values():
                    planner.update_info(copy.deepcopy(info))
                termination = self._termination(done, info, episode)
                if termination:
                    episode['termination'] = termination
                if termination or not succeeded:
                    break
            summary = summarize_reference_execution(reference_trace)
            for record in records:
                episode['generated_examples'].append(dict(modality=record['modality'],
                    trajectory=record['trajectory'], model_failure_code=record['failure_code']))
            # The original generator exports these text files in one episode
            # directory; collected images remain in the existing image source.
            for name, value in [('example_discrete_actions', discrete),
                                ('example_continuous_actions', continuous),
                                ('example_pre_defined_actions', summary['skills']),
                                ('example_reference_object_list', self.env.task._task.reference_objects)]:
                (path / (name + '.txt')).write_text(repr(plain(value)) + '\n')
            (path / 'example_reasoning.txt').write_text('\n\n'.join(
                'reasoning_' + x['trajectory'].replace('-', '_') + '_' +
                ('vlm_icl' if x['modality'] == 'vlm_ablation' else x['modality']) + ':\n' +
                x.get('response', x.get('error', '')) for x in records))
            skill_prompt = next((x['task_prompt'] for x in records
                                 if x['modality'] == 'vlm' and x['trajectory'] == 'pre-defined'), '')
            pose_prompt = next((x['task_prompt'] for x in records
                                if x['modality'] == 'vlm' and x['trajectory'] == 'point'), '')
            (path / 'example_task_prompt.txt').write_text(
                'task_prompt_pre_defined\n' + skill_prompt + '\n\ntask_prompt_point\n' + pose_prompt + '\n')
            first_record = records[0]
            (path / 'example_detection.txt').write_text(
                'avg_obj_coord\n' + repr(first_record['coordinates']) +
                '\nall_avg_point_list\n' + repr(first_points) +
                '\nimplicit_name_to_real_name\n' + repr(first_record['object_names']) + '\n')
            if not episode.get('termination'):
                episode['termination'] = self._termination(False, {}, episode)
            turn += 1


def main():
    parser = argparse.ArgumentParser(description='Generate task examples from reference scenes.')
    parser.add_argument('--model_name', default='gpt-4o-mini')
    parser.add_argument('--model_type', default='remote')
    parser.add_argument('--eval_sets', default='debug', type=lambda value: value.split(','))
    for name, default in [('language_only',1), ('chat_history',0), ('n_shots',1), ('multiview',0),
                          ('detection_box',1), ('multistep',0), ('resolution',500), ('visual_icl',0),
                          ('cognition',1), ('tp',1), ('headless',0), ('text_prompt_with_semantic',1)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--down_sample_ratio', type=float, default=1.0)
    parser.add_argument('--exp_name', default='baseline')
    parser.add_argument('--traj_type', default='pre-defined')
    parser.add_argument('--instruction_type', default='attack', choices=['normal','attack','defense'])
    parser.add_argument('--example_classification_type', default='task')
    # Match the original evaluator's shared dataset and logging controls.
    parser.add_argument('--data_root')
    parser.add_argument('--expected_episodes', type=int, default=100)
    parser.add_argument('--selected_indexes', type=lambda value: [int(i) for i in value.split(',') if i], default=[])
    parser.add_argument('--output_dir')
    parser.add_argument('--max_episode_steps', type=int, default=15)
    args = parser.parse_args()
    config = vars(args)
    config['bbox_source'] = 'yolo'
    evaluator = RRB_ManipulationEvaluator(config)
    try:
        evaluator.check_config_valid()
        evaluator.evaluate_main()
    finally:
        if evaluator.env is not None:
            evaluator.env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
