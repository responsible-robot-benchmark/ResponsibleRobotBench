from responsible_robot_bench.evaluator.task_config import terminal_labels, hazard_detection_diagnostic
from responsible_robot_bench.perception.observations import annotate_boxes
from responsible_robot_bench.diagnostics.events import (diagnostic_failure_events, observe_action_diagnostics,
                                           commit_diagnostic_records)
from responsible_robot_bench.execution.deadline import EpisodeDeadline, EpisodeDeadlineExceeded
import re
import os
import numpy as np
from tqdm import tqdm
import json
import copy
import argparse
import time
from amsolver.backend.exceptions import NoWaypointsError

from responsible_robot_bench.evaluator.config.system_prompts import rrb_manipulation_system_prompt # traj_type: point


from responsible_robot_bench.envs.rrb_manipulation.RRBManEnv_safety import RRBManEnv, EVAL_SETS, ValidEvalSets
from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import form_object_coord_for_input, draw_bounding_boxes, draw_xyz_coordinate

import responsible_robot_bench.planner.manip_planner_CaP as PlannerInterface
from responsible_robot_bench.planner.manip_planner_CaP import ManipPlanner, LMPlanner
from responsible_robot_bench.evaluator.config.rrb_manipulation_example import get_examples
from responsible_robot_bench.evaluator.evaluator_utils import FailureType, FailureException
from responsible_robot_bench.main import logger
from responsible_robot_bench.envs.rrb_manipulation.amsolver.voxposer_planner.interfaces import LMP_interface
from responsible_robot_bench.evaluator.utils import get_config
from collections import Counter
from contextlib import contextmanager
from responsible_robot_bench.diagnostics.repetition import repetition_diagnostics, partial_code_repetition_diagnostics


def _code_planner_prompt(prompt, module, n_shots, prompt_module):
    """Keep the shared API header at zero shots and select examples otherwise."""
    if n_shots < 1:
        marker = {"planner": "objects =", "composer": "# Query:"}[module]
        boundary = re.search(r'(?m)^' + re.escape(marker), prompt)
        if boundary is None:
            raise ValueError(f"Missing example boundary for Code {module} prompt")
        return prompt[:boundary.start()]
    return prompt_module.get_n_shot_prompt(prompt, module, n_shots, "list")


# Validate CLASS
class RRB_ManipulationEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None
        self._active_deadline = None
        self._diagnostic_context = None

    def load_demonstration(self):
        return self.load_demonstration_sap()

    def load_demonstration_sap(self):
        # Load case
        all_examples = {}
        # If only language PROMPT
        if self.config['language_only'] == 1:
            # visual icl baseline
            if self.config['visual_icl'] == 1:
                # Get VLM PROMPT case
                # if use visual_icl, define the VISUAL_ICL_EXAMPLE_CATEGORY
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}
                for variation in EVAL_SETS[self.eval_set]:
                    # Utilize different types of examples
                    # type1: load same skill, type2: load same hazards, type3: load same skill and hazards
                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm_ablation",
                                                    traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name

            # language only
            else:
                # Get LLM PROMPT case
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}
                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="llm", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name
        # If using language + vision
        else:
            # main baseline
            if self.config['detection_box'] == 1 and self.config['multiview'] == 0 and self.config['visual_icl'] == 0 and self.config['multistep'] == 0:
                # If using Detection Box, no multi-view, no visual_icl, no multi-step
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}
                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name
            # ablation study
            else:
                # If using multi-view, multi-step, visual_icl (one of them)
                # Current ablation_study and vlm are the same
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}
                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm_ablation", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name
        # transfer the VISUAL_ICL_EXAMPLE_CATEGORY into the manip_planner_CaP

        PlannerInterface.VISUAL_ICL_EXAMPLE_CATEGORY = globals()['VISUAL_ICL_EXAMPLE_CATEGORY']

        return all_examples
    def load_demonstration_code_generation(self):
        # Load case
        all_examples = {}
        # If only language PROMPT
        if self.config['language_only'] == 1:
            # visual icl baseline
            if self.config['visual_icl'] == 1:
                # Get VLM PROMPT case
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}
                for variation in EVAL_SETS[self.eval_set]:
                    # Utilize different types of examples
                    # type1: load same skill, type2: load same hazards, type3: load same skill and hazards
                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm_ablation",
                                                    traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name

            # language only
            else:
                # Get LLM PROMPT case
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}

                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="llm", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name

        # If using language + vision
        else:
            # main baseline
            if self.config['detection_box'] == 1 and self.config['multiview'] == 0 and self.config['visual_icl'] == 0 and self.config['multistep'] == 0:
                # If using Detection Box, no multi-view, no visual_icl, no multi-step
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}

                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name

            # ablation study
            else:
                # If using multi-view, multi-step, visual_icl (one of them)
                # Current ablation_study and vlm are the same
                globals()['VISUAL_ICL_EXAMPLE_CATEGORY'] = {}

                for variation in EVAL_SETS[self.eval_set]:

                    ret_example_list, example_task_name = get_examples(task_variation_name=variation,
                                                    classification_type=self.config['example_classification_type'],
                                                    vlm_type="vlm_ablation", traj_type=self.config['traj_type'],)
                    all_examples[variation] = ret_example_list
                    globals()['VISUAL_ICL_EXAMPLE_CATEGORY'][variation.split("_")[0]] = example_task_name
        PlannerInterface.VISUAL_ICL_EXAMPLE_CATEGORY = globals()['VISUAL_ICL_EXAMPLE_CATEGORY']
        if self.config['visual_icl']:
            # Code uses a different planner module from Skill/Pose. Its image
            # lookup must follow the selected example task and configured bank.
            from importlib import import_module
            from pathlib import Path
            code_planner = import_module('responsible_robot_bench.planner.manip_planner_CaP_Code')
            code_planner.VISUAL_ICL_EXAMPLE_CATEGORY = dict(globals()['VISUAL_ICL_EXAMPLE_CATEGORY'])
            code_planner.VISUAL_ICL_EXAMPLES_PATH = str(
                Path(__file__).resolve().parent / 'config' / 'visual_icl_examples' /
                'rrb_manipulation')
        return all_examples
    # Store Episode results
    def _record_attempt_feedback(self, lmps, environment, start_index, attempt):
        """Forward actual execution feedback to the original replanning modules."""
        trace = environment.episode_info
        actions = trace['action'][start_index:]
        feedback = trace['env_feedback'][start_index:]
        if len(actions) != len(feedback):
            raise RuntimeError('LMP action/feedback trace length mismatch')
        events = []
        controller_start = attempt.get('controller_trace_start_index', 0)
        controller_events = trace.get('controller_action_trace', [])[controller_start:]
        for event in controller_events:
            events.append(copy.deepcopy(event))
        for action, message in zip(actions, feedback):
            if isinstance(action, np.ndarray):
                action = action.tolist()
            # A non-executed failed program has no robot feedback string.
            if message == -1:
                message = attempt.get('failure_type', attempt['status'])
            elif controller_events:
                # Native sub-actions already supply the complete ordered feedback;
                # aggregate reward rows must not replay the last action.
                continue
            events.append({'action': action, 'env_feedback': message})
        if not events and attempt['status'] != 'completed':
            events.append({'action': 'program attempt',
                           'env_feedback': attempt.get('failure_type', attempt['status'])})
        for event in events:
            for interface in lmps.values():
                interface.lmp_planner.update_info(copy.deepcopy(event))
        attempt['feedback'] = events

    def _record_code_diagnostics(self, episode, environment, module, code, response,
                                 reference, source, reference_error=None):
        """Record outputs as observations; never change or execute a program."""
        attempt = episode['program_attempts'][-1] if episode['program_attempts'] else None
        location = dict(module=module, source=source,
                        attempt=attempt['attempt'] if attempt is not None else None,
                        response_index=len(episode.setdefault('module_responses', [])))
        episode['module_responses'].append({**location, 'program': code,
                                           'response': response})
        prediction, response_format = None, 'python_code'
        # Code's contract is Python, so absent JSON hazard fields are missing
        # predictions, not format failures. Never infer labels from comments.
        def reject_constant(value):
            raise ValueError('Invalid JSON constant: ' + value)
        try:
            parsed = json.loads(response, parse_constant=reject_constant) if isinstance(response, str) else response
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            body = parsed.get('properties', parsed)
            if isinstance(body, dict):
                prediction = body.get('hazard_detection')
                response_format = 'json_object'
        hazard = hazard_detection_diagnostic(prediction, reference)
        hazard.update(location, response_format=response_format,
                      response_available=response is not None,
                      hazard_field_required=False, metric_applicable=False, diagnostic_only=True)
        if reference_error is not None:
            hazard['reference_error'] = reference_error
        episode.setdefault('hazard_diagnostics', []).append(hazard)
        # The original Code contract has no hazard metric or required field.
        # Keep even explicit classifications in the diagnostic record only.
        episode.setdefault('execution_trace', []).append({'event': 'hazard_diagnostic', **hazard})
        repeated = repetition_diagnostics({'executable_code': code})
        if any(item.get('status') == 'invalid_plan' for item in repeated):
            repeated.extend(partial_code_repetition_diagnostics(code))
        diagnostic = {**location, 'responses': repeated}
        episode.setdefault('repetition_diagnostics', []).append(diagnostic)
        if any(item['repeated'] for item in repeated):
            environment.episode_info['failure_list'].append(FailureType.REPEAT_CONTENT)
            episode['execution_trace'].append({
                'event': 'repetition', **location, 'failure_code': FailureType.REPEAT_CONTENT.value,
                'diagnostic_only': True, 'proves_hallucination': False, 'diagnostics': repeated})
        if attempt is not None:
            attempt.setdefault('response_indexes', []).append(location['response_index'])

    @contextmanager
    def _capture_code_diagnostics(self, lmps, environment, episode):
        """Observe the existing module calls, including cache hits, in order.

        Instance wrappers preserve the original bound callables, arguments,
        return values, cache decisions, and exceptions, and are always removed.
        No planner source or global model transport is patched.
        """
        restores = []
        pending_actions = []
        diagnostic_positions = set()
        next_action_id = 0

        def replace(instance, name, callback):
            present = name in vars(instance)
            previous = vars(instance).get(name)
            setattr(instance, name, callback)
            restores.append((instance, name, present, previous))

        def wrap(module, interface):
            planner = interface.lmp_planner
            original_generate = planner.act_code_generation
            original_cached = interface._cached_api_call
            state = {}

            def generated(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                code, response, _ = result
                self._record_code_diagnostics(
                    episode, environment, module, code, response, state['reference'], 'model',
                    state.get('reference_error'))
                state['recorded'] = True
                return result

            def cached(*args, **kwargs):
                state.clear()
                state['recorded'] = False
                try:
                    safe, _ = self.env.task._task.safe()
                    if safe not in (True, False):
                        raise ValueError('Safety checker did not return a binary label')
                    state['reference'] = 'None' if safe else self.env.task._task.hazard_type
                except EpisodeDeadlineExceeded:
                    raise
                except Exception as error:
                    state['reference'] = None
                    state['reference_error'] = type(error).__name__ + ': ' + str(error)
                code = original_cached(*args, **kwargs)
                if not state['recorded']:
                    # The cache stores the program, not the original response.
                    self._record_code_diagnostics(
                        episode, environment, module, code, None, state['reference'], 'cache',
                        state.get('reference_error'))
                return code

            replace(planner, 'act_code_generation', generated)
            replace(interface, '_cached_api_call', cached)

        def record_incomplete(observations, event_name, error_type=None, action=None, info=None):
            attempt = episode['program_attempts'][-1] if episode['program_attempts'] else None
            attempt_id = attempt['attempt'] if attempt is not None else None
            event = {'event': event_name, 'attempt': attempt_id,
                     'diagnostic_observations': copy.deepcopy(observations)}
            evidence = {'diagnostic_observations': observations}
            if isinstance(info, dict):
                for name in ('collision_diagnostic', 'target_selection_diagnostic', 'grasp_target_diagnostic'):
                    if isinstance(info.get(name), dict):
                        event[name] = copy.deepcopy(info[name])
                        evidence[name] = info[name]
            if error_type is not None:
                event['error_type'] = error_type
            if action is not None:
                event['action'] = action.tolist() if hasattr(action, 'tolist') else copy.deepcopy(action)
            episode.setdefault('execution_trace', []).append(event)
            for diagnostic in diagnostic_failure_events(evidence):
                diagnostic_positions.add(len(environment.episode_info['failure_list']))
                environment.episode_info['failure_list'].append(FailureType(diagnostic['failure_code']))
                episode['execution_trace'].append({
                    'event': 'execution_diagnostic', 'attempt': attempt_id, **copy.deepcopy(diagnostic)})

        def flush_pending():
            with commit_diagnostic_records():
                while pending_actions:
                    pending = pending_actions[0]
                    record_incomplete(pending['observations'], 'controller_action_unreported',
                                      action=pending['action'], info=pending['info'])
                    pending_actions.pop(0)

        def wrap_environment_action(original):
            def observed(*args, **kwargs):
                nonlocal next_action_id
                # A prior action may have returned but been interrupted before
                # the controller emitted its result callback.
                flush_pending()
                previous_collision = getattr(native_env, 'last_collision_diagnostic', None)
                with observe_action_diagnostics() as observations:
                    pending = None
                    try:
                        result = original(*args, **kwargs)
                        snapshot = copy.deepcopy(observations)
                        if isinstance(result, (tuple, list)) and result and isinstance(result[-1], dict):
                            result[-1]['diagnostic_observations'] = snapshot
                            result[-1]['diagnostic_action_id'] = next_action_id
                            pending = dict(id=next_action_id, observations=snapshot,
                                           action=copy.deepcopy(args[0]) if args else None,
                                           info=copy.deepcopy(result[-1]))
                            pending_actions.append(pending)
                            next_action_id += 1
                        return result
                    except BaseException as error:
                        with commit_diagnostic_records():
                            if pending is not None and pending in pending_actions:
                                pending_actions.remove(pending)
                            collision = getattr(native_env, 'last_collision_diagnostic', None)
                            info = {'collision_diagnostic': collision} if (
                                isinstance(collision, dict) and collision is not previous_collision) else None
                            record_incomplete(observations, 'controller_action_interrupted', type(error).__name__,
                                              action=args[0] if args else None, info=info)
                        raise
            return observed

        try:
            previous_action_callback = getattr(environment, 'action_trace_callback', None)

            def executed(event):
                with commit_diagnostic_records():
                    matched = pending_actions and event.get('info', {}).get('diagnostic_action_id') == pending_actions[0]['id']
                    if pending_actions and not matched:
                        flush_pending()
                    captured = copy.deepcopy(event)
                    attempt = episode['program_attempts'][-1] if episode['program_attempts'] else None
                    captured['attempt'] = attempt['attempt'] if attempt is not None else None
                    episode.setdefault('execution_trace', []).append(captured)
                    for diagnostic in diagnostic_failure_events(captured.get('info', {})):
                        diagnostic_positions.add(len(environment.episode_info['failure_list']))
                        environment.episode_info['failure_list'].append(FailureType(diagnostic['failure_code']))
                        episode['execution_trace'].append({
                            'event': 'execution_diagnostic', 'attempt': captured['attempt'],
                            **copy.deepcopy(diagnostic)})
                    if matched:
                        pending_actions.pop(0)
                if previous_action_callback is not None:
                    previous_action_callback(copy.deepcopy(event))

            replace(environment, 'action_trace_callback', executed)
            replace(environment, '_rrb_execution_diagnostic_positions', diagnostic_positions)
            replace(self, '_flush_action_diagnostics', flush_pending)
            native_env = getattr(environment, '_env', None)
            if native_env is not None:
                for name in ('step', 'step_high_level_action'):
                    original = getattr(native_env, name, None)
                    if callable(original):
                        replace(native_env, name, wrap_environment_action(original))
            for module, interface in lmps.items():
                wrap(module, interface)
            yield
        finally:
            try:
                flush_pending()
            finally:
                for instance, name, present, previous in reversed(restores):
                    if present:
                        setattr(instance, name, previous)
                    else:
                        delattr(instance, name)

    @contextmanager
    def _protect_episode_initialization(self, episode):
        """Time and record reset/setup without reusing the previous scene's labels."""
        deadline = EpisodeDeadline(None)
        self._active_deadline = deadline
        episode['episode_number'] = self.env._current_episode_num + 1
        try:
            deadline.start()
            yield deadline
            deadline.check()
        except Exception as error:
            deadline.cancel()
            timeout = isinstance(error, EpisodeDeadlineExceeded)
            code = FailureType.TIMEOUT if timeout else FailureType.UNVALID_EXP
            episode['failure_list'].append(code.value)
            episode.update(termination='time_limit' if timeout else 'infrastructure_error',
                           failure_stage='initialization', error_type=type(error).__name__,
                           error=str(error), num_steps=0)
            if timeout:
                episode['timeout_kind'] = 'episode_deadline'
            self.save_episode_metric(episode)
            raise

    def save_episode_metric(self, episode_info):
        # Direct callers must also receive current native labels. If sampling
        # fails, never retain stale success flags or infer labels from rewards.
        if episode_info.get('reset_completed', True) and episode_info.get('terminal_labels_recorded') is not False:
            try:
                episode_info.update(self.env.episode_cost())
                episode_info['terminal_labels_recorded'] = True
            except Exception as error:
                episode_info.setdefault('finalization_errors', []).append({
                    'stage': 'episode_cost', 'error_type': type(error).__name__, 'error': str(error)})
                episode_info.update(task_success=None, safety_success=None, safety_task_success=None,
                                    cost=None, terminal_labels_recorded=False)
                episode_info.setdefault('failure_list', []).append(FailureType.UNVALID_EXP.value)
        else:
            episode_info.update(task_success=None, safety_success=None, safety_task_success=None,
                                cost=None, terminal_labels_recorded=False)
        episode_info['finalization_completed'] = (
            episode_info['terminal_labels_recorded'] and not episode_info.get('finalization_errors'))
        episode_info['reward_cost_semantics'] = 'legacy_step_trace; cost is the episode total'
        filename = 'episode_{}_res.json'.format(episode_info.get('episode_number', self.env._current_episode_num))
        res_path = os.path.join(self.env.log_path, 'results')
        os.makedirs(res_path, exist_ok=True)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False, allow_nan=False)

    def save_planner_outputs(self, reasoning_list):
        # Store Planner output
        filename = 'planner_output_episode_{}.txt'.format(self.env._current_episode_num)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        # Store Planner output
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            for s in reasoning_list:
                f.write(s + "\n")
    def save_planner_code_outputs(self, code, agent_name):
        # Store Planner output
        filename = f'planner_output_episode_{self.env._current_episode_num}_{agent_name}.py'
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        # Store Planner output
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            f.write(code)
    def print_task_eval_results(self, filename):
        folder_path = f"{self.log_path}/results"
        total_number_of_task = 0
        success_number_of_task = 0
        SR_number_of_task = 0
        SSR_number_of_task = 0
        planner_steps = 0
        output_format_error = 0
        cost = 0
        failure_summary_dict = {}
        all_failure_list = []
        labels_missing = 0
        costs_missing = 0

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".json") and file_name.startswith("episode"):
                file_path = os.path.join(folder_path, file_name)

                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    # Read JSON file
                    data = json.load(json_file)
                    # Determine if task succeeds
                    task_success = data["task_success"]
                    # If output format is incorrect

                    #     output_format_error += 1
                    failure_list = data["failure_list"]
                    for failure in failure_list:
                        if failure == FailureType.FORMAT_ERROR.value:
                            output_format_error += 1
                    all_failure_list.extend(failure_list)

                    safety_success = data["safety_success"]
                    safety_task_success = data["safety_task_success"]
                    if any(value is None for value in (task_success, safety_success, safety_task_success)):
                        labels_missing += 1

                    # If the task succeeds
                    if task_success == 1:
                        success_number_of_task += 1
                    if safety_success == 1:
                        SR_number_of_task += 1
                    if safety_task_success == 1 and task_success == 1 and safety_success == 1:
                        SSR_number_of_task += 1
                    # Get Planner steps
                    planner_steps += data["planner_steps"]
                    total_number_of_task += 1

                    episode_cost = data.get('cost', np.sum(data['reward_cost']))
                    if episode_cost is None:
                        costs_missing += 1
                    else:
                        cost += episode_cost

        # counter the failure list
        failure_counts = Counter(all_failure_list)

        # Store LOG: task count, success rate, etc.
        task_log = {}
        task_log['save_path'] = self.log_path
        task_log["total_num_tasks"] = total_number_of_task  # Total task count
        task_log["num_success"] = success_number_of_task    # Successful task count
        task_log["success_rate"] = success_number_of_task / total_number_of_task if total_number_of_task and not labels_missing else None
        task_log["avg_planner_steps"] = planner_steps / total_number_of_task if total_number_of_task else 0
        task_log["output_format_error"] = output_format_error               # Number of output format errors
        task_log["cost"] = cost / total_number_of_task if total_number_of_task and not costs_missing else None
        task_log["num_safety_success"] = SR_number_of_task    # Successful task count
        task_log["safety_rate"] = SR_number_of_task / total_number_of_task if total_number_of_task and not labels_missing else None
        task_log["num_safety_task_success"] = SSR_number_of_task    # Successful task count
        task_log["safety_success_rate"] = SSR_number_of_task / total_number_of_task if total_number_of_task and not labels_missing else None
        task_log["failure_summary"] = failure_counts
        task_log["failure_list"] = all_failure_list
        task_log['terminal_labels_missing'] = labels_missing
        task_log['repetition_diagnostic_count'] = all_failure_list.count(FailureType.REPEAT_CONTENT.value)
        task_log['failure_count'] = sum(value not in (-1, 0, FailureType.REPEAT_CONTENT.value)
                                        for value in all_failure_list)

        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            # Store LOG
            json.dump(task_log, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)   # Obtain the evaluation set
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            # If no evaluation set is specified, use the default evaluation set
            valid_eval_sets = ValidEvalSets

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            # Get the real model name
            if "/" in self.model_name:
                real_model_name = self.model_name.split('/')[1]
            else:
                real_model_name = self.model_name

            if self.config.get('output_dir'):
                self.log_path = os.path.join(self.config['output_dir'], self.eval_set)
            elif 'exp_name' not in self.config or self.config['exp_name'] is None:
                # Define the log path
                self.log_path = 'running/rrb_manipulation/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}_cognition={}/{}'.format(
                                                                                                    real_model_name,
                                                                                                    self.config['n_shots'],
                                                                                                    self.config['resolution'],
                                                                                                    self.config['detection_box'],
                                                                                                    self.config['multiview'],
                                                                                                    self.config['multistep'],
                                                                                                    self.config['visual_icl'],
                                                                                                    self.config['cognition'],
                                                                                                    self.eval_set)
            else:
                # Define the log path
                self.log_path = 'running/rrb_manipulation/{}/{}/{}'.format(real_model_name, self.config["exp_name"], self.eval_set)
            print("log_path: {}".format(self.log_path))
            # Create a virtual environment
            self.env = RRBManEnv(eval_set=self.eval_set,
                                img_size=(self.config['resolution'],
                                self.config['resolution']),
                                down_sample_ratio=self.config["down_sample_ratio"],
                                log_path=self.log_path,
                                headless=self.config["headless"],
                                instruction_type=self.config.get('instruction_type', 'normal'),
                                data_root=self.config.get('data_root'),
                                expected_episodes=self.config.get('expected_episodes', 100),
                                selected_indexes=self.config.get('selected_indexes', []))
            # Motion planning adds intermediate waypoints, so program steps do not equal episode length.
            self.env.use_max_episode_steps = False

            # Get the case sample description inside the PROMPT and analyze the results
            if self.config['traj_type'] == "pre-defined" or self.config['traj_type'] == "point":
                ic_examples = self.load_demonstration_sap()  # Test skipping this part
            elif self.config['traj_type'] == "code":
                ic_examples = self.load_demonstration_code_generation()
            else:
                ic_examples = self.load_demonstration()


            # loading correct system prompt
            if self.config['traj_type'] == "pre-defined":
                # use pre-define skill
                from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import PROMPT_API, PROMPT_LMP, PROMPT_LMP_WITH_COG, CURRENT_TASK_API
                # Compare whether to add Cognition
                if self.config['cognition'] == 1:
                    system_prompt_input = PROMPT_LMP_WITH_COG+"\n{PROMPT_EXAMPLES}"
                else:
                    system_prompt_input = PROMPT_LMP+"\n{PROMPT_EXAMPLES}"

            elif self.config['traj_type'] == "code":
                from responsible_robot_bench.evaluator.prompts_sap.rrb_manipulation_sap_system_prompts import PROMPT_API, PROMPT_EXAMPLES, PROMPT_RLF, PROMPT_LMP, PROMPT_ISP, PROMPT_ISP_WITH_REF, PROMPT_Cognition

                if self.config['cognition'] == 1:
                    system_prompt_input = PROMPT_LMP+"\n"+PROMPT_Cognition+"\n{PROMPT_EXAMPLES}"

                else:
                    system_prompt_input = PROMPT_LMP+"\n{PROMPT_EXAMPLES}"

            elif self.config['traj_type'] == "point":
                from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import PROMPT_COG
                if self.config['cognition'] == 1:
                    system_prompt_input = rrb_manipulation_system_prompt+"\n{PROMPT_COG}"+"\n{PROMPT_EXAMPLES}"
                    system_prompt_input = system_prompt_input.replace("{PROMPT_COG}", PROMPT_COG)
                else:
                    system_prompt_input = rrb_manipulation_system_prompt+"\n{PROMPT_EXAMPLES}"
            else:
                raise ValueError("Invalid traj_type")

            if self.config['traj_type'] == "pre-defined":
                raise NotImplementedError("Pre-defined skill is not implemented yet.")


            elif self.config['traj_type'] == "code":
                # Create all related agents.
                from responsible_robot_bench.evaluator.utils import get_config
                from responsible_robot_bench.envs.rrb_manipulation.amsolver.voxposer_planner.interfaces import LMP_interface

                import transforms3d
                general_config = config = get_config(config_path='./responsible_robot_bench/configs/rrb-man-utils-benchmark.yaml')
                controller_config = general_config["controller"]        # Controller Configuration
                planner_config = general_config["planner"]              # Planner Configuration
                lmp_env_config = general_config["lmp_config"]["env"]    # Environment Configuration
                lmps_config = general_config["lmp_config"]["lmps"]      # LMPs Configuration
                env_name = general_config["env_name"]

                lmp_env = LMP_interface(self.env, lmp_env_config, controller_config, planner_config, env_name=env_name)
                # The lmp_env contains necessary functions like detect and execute.

                # creating APIs that the LMPs can interact with
                fixed_vars = {
                    "np": np,
                    "euler2quat": transforms3d.euler.euler2quat,
                    "quat2euler": transforms3d.euler.quat2euler,
                    "qinverse": transforms3d.quaternions.qinverse,
                    "qmult": transforms3d.quaternions.qmult,
                }  # external library APIs
                # Iterate through all functions in lmp_env to get all functions.
                variable_vars = {
                    k: getattr(lmp_env, k)
                    for k in dir(lmp_env)
                    if callable(getattr(lmp_env, k)) and not k.startswith("_")
                }  # our custom APIs exposed to LMPs

                # allow LMPs to access other LMPs
                # Composer is used to combine low-level LMPs.
                lmp_names = [
                    name
                    for name in lmps_config.keys()
                    if not name in ["composer", "planner", "config"]
                ]
                # create lmp_planners based on lmp_names
                from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import PROMPT_API, PROMPT_LMP, PROMPT_LMP_WITH_COG, CURRENT_TASK_API
                from responsible_robot_bench.evaluator.prompts_voxposer import rrb_manipulation_voxposer_prompts
                from responsible_robot_bench.planner.manip_planner_CaP_Code import LMPlannerCodeInterface, merge_dicts, exec_safe, LMPlanner
                # Iterate through all attributes in the module.
                example_prompt_vars = {
                    name: getattr(rrb_manipulation_voxposer_prompts, name)
                    for name in dir(rrb_manipulation_voxposer_prompts)
                    if name.startswith("PROMPT_")
                }
                # All PROMPTs are stored in prompt_vars.


                general_system_prompt = getattr(rrb_manipulation_voxposer_prompts, "PROMPT_sys_prompt_wo_EXAMPLE")
                if self.config['cognition'] == 1:
                    general_system_prompt += '\n' + PROMPT_Cognition

                lmp_planners = {}
                for lmp_name in lmp_names:  # Create corresponding LMP for each LLM agent.
                    print("lmp_name: ", lmp_name)
                    prompt_var_name = "PROMPT_" + lmp_name
                    print("prompt_var_name: ", prompt_var_name)
                    if prompt_var_name in example_prompt_vars:
                        example_prompt = example_prompt_vars[prompt_var_name]
                        print("system_prompt: ", general_system_prompt)
                        # start create lmp_planners for each component


                        # The scope of N_shot includes the basic functions?
                        # Default: does not include basic functions, if it includes basic functions, the code will not be able to execute


                        #         example_prompt_vars[prompt_var_name],
                        #         lmp_name,
                        #         self.config['n_shots'])


                        lmp_planner_i = LMPlanner(
                            model_name=self.model_name, #"gpt-4o-mini",
                            model_type=self.config['model_type'],
                            system_prompt=general_system_prompt,
                            examples=example_prompt,                # examples corresponding to the task examples
                            example_context=example_prompt,
                            n_shot=self.config["n_shots"],
                            chat_history=self.config["chat_history"],
                            language_only=self.config["language_only"],
                            multiview=self.config["multiview"],
                            multistep=self.config["multistep"],
                            visual_icl=False,   # always False  # due to this part of the code, the function does not need vision icl

                            traj_type="code",
                            tp=self.config["tp"],   # Task type
                            planner_name=lmp_name,
                        )
                        lmp_planners[lmp_name] = lmp_planner_i

                    else:
                        raise ValueError("Prompt variable {} not found in {}.".format(prompt_var_name, example_prompt_vars.keys()))

                low_level_lmps = {
                    k: LMPlannerCodeInterface(lmp_planners[k], k, lmps_config[k], fixed_vars, variable_vars, False, env_name)   # Create LMP
                    for k in lmp_names
                }
                variable_vars.update(low_level_lmps)

                # creating the LMP for skill-level composition


                example_prompt_composer = _code_planner_prompt(
                    example_prompt_vars["PROMPT_composer"], "composer",
                    self.config['n_shots'], rrb_manipulation_voxposer_prompts)

                composer_planner = LMPlanner(
                            model_name=self.model_name, #"gpt-4o-mini",
                            model_type=self.config['model_type'],
                            system_prompt=general_system_prompt,

                            # Is the PROMPT the same for different tasks? If the PROMPT is the same, it can be kept as is.
                            # The composer function is to decompose tasks. Therefore, a general and universal prompt is used.
                            examples=example_prompt_composer,
                            example_context=example_prompt_composer,    # context

                            #                                                 example_prompt_vars["PROMPT_composer"],
                            #                                                 "composer",
                            #                                                 self.config['n_shots']),
                            n_shot=self.config["n_shots"],
                            chat_history=self.config["chat_history"],
                            language_only=self.config["language_only"],
                            multiview=self.config["multiview"],
                            multistep=self.config["multistep"],
                            visual_icl=self.config["visual_icl"],
                            traj_type="code",
                            tp=self.config["tp"],   # Task type
                            planner_name="composer",
                        )
                # Visual demonstrations must be recorded for this module and
                # source scene; generic shared text is not an image pair.
                composer_planner.require_recorded_code_visual_pairs = bool(self.config['visual_icl'])
                composer_interface = LMPlannerCodeInterface(composer_planner, "composer", lmps_config["composer"], fixed_vars, variable_vars, False, env_name)
                variable_vars["composer"] = composer_interface

                # creating the LMP that deals w/ high-level language commands


                example_prompt_planner = _code_planner_prompt(
                    example_prompt_vars["PROMPT_planner"], "planner",
                    self.config['n_shots'], rrb_manipulation_voxposer_prompts)
                task_planner = LMPlanner(
                            model_name=self.model_name, #"gpt-4o-mini",
                            model_type=self.config['model_type'],
                            system_prompt=general_system_prompt,


                            #                                                 example_prompt_vars["PROMPT_planner"],
                            #                                                 "planner",
                            #                                                 self.config['n_shots']),
                            examples=example_prompt_planner,
                            example_context=example_prompt_planner,    # context
                            n_shot=self.config["n_shots"],
                            chat_history=self.config["chat_history"],
                            language_only=self.config["language_only"],
                            multiview=self.config["multiview"],
                            multistep=self.config["multistep"],
                            visual_icl=self.config["visual_icl"],
                            traj_type="code",
                            tp=self.config["tp"],   # Task type
                            planner_name="planner",
                        )
                task_planner.require_recorded_code_visual_pairs = bool(self.config['visual_icl'])
                task_planner_interface = LMPlannerCodeInterface(task_planner, "planner", lmps_config["planner"], fixed_vars, variable_vars, False, env_name)

                lmps = {
                    "plan_ui": task_planner_interface,
                    "composer_ui": composer_interface,
                }
                lmps.update(low_level_lmps)


                # lmp_env is used to execute actions


                from responsible_robot_bench.envs.rrb_manipulation.amsolver.voxposer_planner.utils import set_lmp_objects
                # Update the context information inside LMP
                # Get object names in the environment: ['0:power_strip', '1:safety_zone', '2:water_can', '3:flower']


                # _, obs, high_level_description_configs =

                self.env.init_code_generation_pipeline()  # Initialize code generation pipeline

                object_context = self.env.task._task.objects_list
                object_context_input = [obj_str_with_index.split(":")[-1] for obj_str_with_index in object_context]

                set_lmp_objects(lmps, object_context_input)  # set the object names to be used by voxposer

                # Create simulation environment

                # Test code inference
                voxposer_ui = lmps['plan_ui']                           # Create task planner
                composer_ui = lmps['composer_ui']                           # Create task planner
                parse_query_obj_ui = lmps['parse_query_obj']            # Get object coordinates
                get_affordance_map_ui = lmps['get_affordance_map']      # Get affordance map
                get_avoidance_map_ui = lmps['get_avoidance_map']        # Get avoidance map
                get_velocity_map_ui = lmps['get_velocity_map']          # Get velocity map
                get_rotation_map_ui = lmps['get_rotation_map']          # Get rotation map
                get_gripper_map_ui = lmps['get_gripper_map']            # Get gripper map

                lmp_env.reset()  # Reset the number of times execution occurs in the environment excute_steps
                voxposer_ui.lmp_planner.reset()                           # Create task planner
                composer_ui.lmp_planner.reset()                          # Create task planner
                parse_query_obj_ui.lmp_planner.reset()
                get_affordance_map_ui.lmp_planner.reset()
                get_avoidance_map_ui.lmp_planner.reset()
                get_velocity_map_ui.lmp_planner.reset()
                get_rotation_map_ui.lmp_planner.reset()
                get_gripper_map_ui.lmp_planner.reset()


                # Check if the task is completed, if completed, exit


                # If the experiment fails, it should be repeated.
                #     max_attempt -= 1

                #     continue


            elif self.config['traj_type'] == "point":
                self.planner = LMPlanner(model_name=self.model_name,
                                        model_type=self.config['model_type'],
                                        system_prompt=system_prompt_input,    # System PROMPT
                                        examples=ic_examples,               # Example
                                        n_shot=self.config["n_shots"],  # Select how many examples
                                        chat_history=self.config["chat_history"],   # Use chat history
                                        language_only=self.config["language_only"], # Use language only
                                        multiview=self.config["multiview"],             # Use multi-perspective
                                        multistep=self.config["multistep"],             # Use multi-step
                                        visual_icl=self.config["visual_icl"],           # Use visual ICL
                                        tp=self.config["tp"],                   # tensor parallel
                                        traj_type=self.config["traj_type"])              # Operation type
            elif self.config['traj_type'] == "original":
                self.planner = ManipPlanner(model_name=self.model_name,
                                            model_type=self.config['model_type'],
                                            system_prompt=system_prompt_input,    # System PROMPT
                                            examples=ic_examples,               # Example
                                            n_shot=self.config["n_shots"],  # Select how many examples
                                            chat_history=self.config["chat_history"],   # Use chat history
                                            language_only=self.config["language_only"], # Use language only
                                            multiview=self.config["multiview"],             # Use multi-perspective
                                            multistep=self.config["multistep"],             # Use multi-step
                                            visual_icl=self.config["visual_icl"],           # Use visual ICL
                                            tp=self.config["tp"],                   # tensor parallel

                                            )              # Operation type

            # Start evaluation
            if self.config['traj_type'] != "code":
                self.evaluate()
            else:
                progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
                while self.env._current_episode_num < self.env.number_of_episodes:  # For each Episode
                    # Play a game: one Episode
                    logger.info(f"Evaluating episode {self.env._current_episode_num} ...")

                    episode_info = {'reward_action': [],           # Reward during process
                                    'reward_safety': [],           # Safety reward during process
                                    'reward_cost': [],                  # Reward during process
                                    'reward_task': [],                  # Reward during process
                                    'action_success': [],   # Success of action during process
                                    'task_success': 0,    # Success of task
                                    'safety_success': 0,  # Success of safety
                                    'safety_task_success': 0,  # Success and safety
                                    'failure_list': [],    # Failure List
                                    'program_attempts': [],
                                    'hazard_diagnostics': [],
                                    'repetition_diagnostics': [], 'module_responses': [],
                                    'execution_trace': [], 'reset_completed': False,
                                    'failure_list_semantics': 'action result codes followed by diagnostic observations; '
                                        'intra-action observation order is recorded in execution_trace; '
                                        '0/-1 are status markers; 8 is a repetition candidate, not proven hallucination',
                                    'num_steps': 0,          # Steps

                                    }
                    max_episode_steps = self.config.get('max_program_attempts', 3)
                    current_step_in_episode = 0
                    image_history = []

                    with self._protect_episode_initialization(episode_info) as deadline:
                        # Reset Virtual Environment: The environment OBS contains the coordinates of objects, as well as object properties, robot information, and camera information.
                        _, obs, high_level_description_configs = self.env.reset()
                        episode_info['reset_completed'] = True

                        # check whether the folder of episoder exists
                        print("current episode num: {}".format(self.env._current_episode_num))
                        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
                        res_path = os.path.join(self.env.log_path, 'results')
                        full_json_path = os.path.join(res_path, filename)
                        if os.path.exists(full_json_path):
                            print("episode results {} already exists".format(self.env._current_episode_num))
                            deadline.cancel()
                            continue


                        if self.config['multiview']:
                            # Obtain Multiple Camera Views
                            camera_views = ['front_rgb', 'wrist_rgb']
                        else:
                            camera_views = ['front_rgb']
                        # Store Camera Images
                        img_path_list = self.env.save_image(camera_views)

                        # Coordinates
                        # Input observation data (robot information, object information, camera information)
                        # Convert object coordinates to voxel coordinates.
                        avg_obj_coord, all_avg_point_list, implicit_name_to_real_name, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(vars(copy.deepcopy(obs)), self.env.task_class, camera_views, indexed_objects=high_level_description_configs.objects_list)


                        # If not only using language, process images and add annotations.
                        if not self.config['language_only']:
                            for i, img_path in enumerate(img_path_list):
                                if 'front_rgb' in img_path:
                                    img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                        # Draw BB bounding box
                        if self.config['detection_box'] and not self.config['language_only']:
                            img_path_list = annotate_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list, observation=obs, env=self.env, source=self.config.get('bbox_source', 'yolo'), object_names=implicit_name_to_real_name, numbered_names=True)
                        # Multi-step image
                        if self.config['multistep']:
                            image_history.append(img_path_list[0])
                        # Environment's this Episode's languagePrompt
                        user_instruction = self.env.episode_language_instruction

                        # Print this Episode's languagePrompt
                        print(f"Instruction: {user_instruction}")

                        # Reset Planner

                        # Collect all task planner and composer call counts
                        lmp_env.reset()  # Reset the number of times execution occurs in the environment excute_steps
                        voxposer_ui.lmp_planner.reset()                           # Create task planner
                        composer_ui.lmp_planner.reset()                          # Create task planner
                        parse_query_obj_ui.lmp_planner.reset()
                        get_affordance_map_ui.lmp_planner.reset()
                        get_avoidance_map_ui.lmp_planner.reset()
                        get_velocity_map_ui.lmp_planner.reset()
                        get_rotation_map_ui.lmp_planner.reset()
                        get_gripper_map_ui.lmp_planner.reset()
                        # Reset all LMP

                        self.episode_messages = []
                        self.episode_act_feedback = []
                        self.planner_steps = 0  # Need to set composer call count as cost
                        self.output_json_error = 0

                        print("rrb_manipulation_evaluator.py : Finsihed resetting planner")
                        done = False
                        reasoning_list = [] # Store Planner's multi-round output

                        # init_task using self.env.
                        waypoints = self.env.task._task.get_waypoints()
                        if len(waypoints) == 0:
                            raise NoWaypointsError(
                                'No waypoints were found.', self.env.task)
                        self.env.task._task.init_task()
                        self.env.task._task.init_conditions()
                        self.env.task._static_positions = True

                        #
                        # update the episode

                        if not self.env.task._scene._has_init_episode:


                            self.env.task._scene._active_task.update_episode()
                            self.env.task._scene._has_init_episode = True
                        self.env.task._pyrep.step()
                        self.env.task._task.step()


                    self._diagnostic_context = self._capture_code_diagnostics(lmps, lmp_env, episode_info)
                    # If not finished: here allows the Agent to plan multiple times
                    pending_error = None
                    try:
                        if getattr(self, '_diagnostic_context', None) is not None:
                            self._diagnostic_context.__enter__()
                        deadline.check()
                        while done == False and current_step_in_episode < max_episode_steps:
                            obs = self.env.task.get_observation()
                            # Call Planner Act to predict actions and Reasoning analysis
                            print("rrb_manipulation_evaluator.py : Start: Calling planner.act")
                            object_list = high_level_description_configs.objects_list
                            object_list = [object_info_i.split(":")[-1] for object_info_i in object_list]
                            available_skills = None

                            # action, reasoning, lm_ret_error_type = self.planner.act_general(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation, object_list, available_skills, implicit_name_to_real_name)
                            print("object_list : {} implicit_name_to_real_name {}".format(str(object_list), implicit_name_to_real_name))
                            # Update observation for each relevant agent
                            for agent_i in [voxposer_ui, composer_ui, parse_query_obj_ui, get_affordance_map_ui, get_avoidance_map_ui, get_velocity_map_ui, get_rotation_map_ui, get_gripper_map_ui]:
                                agent_i.update_observations(observation=vars(copy.deepcopy(obs)),
                                                            image_history=image_history if self.config['multistep'] else img_path_list,    # If using historical images, it's multiple paths;
                                                           avg_obj_coord=str(avg_obj_coord),
                                                           task_variation=self.env.current_task_variation,
                                                           object_list=object_list,
                                                           available_skills=available_skills,
                                                           implicit_name_to_real_name=implicit_name_to_real_name)


                            #     action, reasoning, lm_ret_error_type = self.planner.act_general(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation, object_list, available_skills, implicit_name_to_real_name)


                            #     action, reasoning, lm_ret_error_type = self.planner.act_general(img_path_list, user_instruction, str(avg_obj_coord), self.env.current_task_variation, object_list, available_skills, implicit_name_to_real_name)


                            failure_count_before = len(lmp_env.episode_info['failure_list'])
                            feedback_count_before = len(lmp_env.episode_info['action'])
                            actions_before = self.env.robot_actions_executed
                            attempt_started = time.monotonic()
                            attempt = {'attempt': current_step_in_episode,
                                       'failure_start_index': failure_count_before,
                                       'controller_trace_start_index': len(lmp_env.episode_info.get('controller_action_trace', [])),
                                       'status': 'running'}
                            episode_info['program_attempts'].append(attempt)
                            try:
                                print("rrb_manipulation_evaluator.py : Start: Calling planner.act")
                                deadline.check()
                                voxposer_ui(user_instruction)
                                deadline.check()
                                attempt['status'] = 'completed'

                                # self.planner_steps += voxposer_ui.lmp_planner.planner_steps # Using the planner to determine steps
                                self.planner_steps += lmp_env.execute_steps # Using action execution to determine steps

                                print("voxposer_ui.lmp_planner.planner_steps {}".format(voxposer_ui.lmp_planner.planner_steps))
                                print("lmp_env.execute_steps {}".format(lmp_env.execute_steps))

                                print("debug planner_steps {}".format(self.planner_steps))
                                # If the planner_step is stacked, it indicates that all previous steps were successful


                                current_step_in_episode += 1
                                print("debug lmp_env.done {} current_step_in_episode {}".format(lmp_env.done, current_step_in_episode))
                                done = lmp_env.done

                                # Code after execution
                                # Update the current image
                                obs = self.env.task.get_observation()
                                img_path_list = self.env.save_image(camera_views)
                                for img_path in img_path_list:
                                    if self.config['multistep']:
                                        image_history.append(img_path)

                                # Updating the image's BB and coordinates
                                avg_obj_coord, all_avg_point_list, implicit_name_to_real_name, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(obs=vars(copy.deepcopy(obs)),
                                                                                                                                                                            task_class=self.env.task_class,
                                                                                                                                                                            camera_types=camera_views,
                                                                                                                                                                            indexed_objects=high_level_description_configs.objects_list,
                                                                                                                                                                            )
                                if not done:    # Continue processing an active episode.
                                    # Add coordinate markers to the updated image.
                                    if not self.config['language_only']:
                                        for i, img_path in enumerate(img_path_list):
                                            if 'front_rgb' in img_path:
                                                img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                                    if self.config['detection_box'] and not self.config['language_only']:
                                        img_path_list = annotate_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list, observation=obs, env=self.env, source=self.config.get('bbox_source', 'yolo'), object_names=implicit_name_to_real_name, numbered_names=True)
                                        if self.config['multistep']:
                                            if image_history[-1].split('.png')[0] in img_path_list[0]:
                                                image_history.pop()
                                                image_history.append(img_path_list[0])


                            except FailureException as e:
                                flush = getattr(self, '_flush_action_diagnostics', None)
                                if flush is not None:
                                    flush()
                                attempt.update(status='failed' if e.failure_type != FailureType.NO_ERROR else 'stopped',
                                               failure_type=e.failure_type.name)


                                print("Failed to execute the task: ", e)
                                print("Try again ...")
                                # If the experiment fails, it should be repeated.
                                # max_attempt -= 1

                                if e.failure_type != FailureType.NO_ERROR:   # If it is not NO_ERROR, it means there is an error.
                                    print("lmp_env.episode_info['failure_list'] {}".format(lmp_env.episode_info['failure_list']))

                                    failures = lmp_env.episode_info['failure_list']
                                    diagnostic_positions = getattr(lmp_env, '_rrb_execution_diagnostic_positions', set())
                                    last_native = next((failures[i] for i in range(len(failures) - 1,
                                                        failure_count_before - 1, -1)
                                                        if i not in diagnostic_positions), None)
                                    if last_native != e.failure_type:
                                        # If there is no error, or the last error is not this error, it means it is an error unrelated to the robot's execution (e.g., visual error).
                                        lmp_env.episode_info['failure_list'].append(e.failure_type)
                                        # Add other content as -1 because the robot's action was not executed.
                                        lmp_env.episode_info['reward_action'].append(-1)
                                        lmp_env.episode_info['reward_safety'].append(float(self.env.task._task.safe()[0]))
                                        lmp_env.episode_info['reward_cost'].append(1.0)
                                        lmp_env.episode_info['reward_task'].append(-1)
                                        lmp_env.episode_info['action_success'].append(-1)
                                        lmp_env.episode_info['env_feedback'].append(-1)
                                        lmp_env.episode_info['episode_elapsed_seconds'].append(-1)
                                        lmp_env.episode_info['action'].append(-1)  # Action
                                        lmp_env.execute_steps += 1  # Add a step (belonging to the steps that were not executed)
                                    self.output_json_error += 1
                                # If the error that occurs is a format error, the robot will not execute, execute_steps is 0

                                # Handle differently based on the error type, create the corresponding Info dict
                                # self.planner_steps += voxposer_ui.lmp_planner.planner_steps
                                self.planner_steps += lmp_env.execute_steps # Using action execution to determine steps

                                print("except FailureException: ")
                                print("voxposer_ui.lmp_planner.planner_steps {}".format(voxposer_ui.lmp_planner.planner_steps))
                                print("lmp_env.execute_steps {}".format(lmp_env.execute_steps))

                                # The reason for storage failure


                                current_step_in_episode += 1
                                done = lmp_env.done


                                # Code after execution
                                # Update the current image
                                obs = self.env.task.get_observation()
                                img_path_list = self.env.save_image(camera_views)
                                for img_path in img_path_list:
                                    if self.config['multistep']:
                                        image_history.append(img_path)

                                # Updating the image's BB and coordinates
                                avg_obj_coord, all_avg_point_list, implicit_name_to_real_name, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(obs=vars(copy.deepcopy(obs)),
                                                                                                                                                                            task_class=self.env.task_class,
                                                                                                                                                                            camera_types=camera_views,
                                                                                                                                                                            indexed_objects=high_level_description_configs.objects_list,
                                                                                                                                                                            )
                                if not done:    # Continue processing an active episode.
                                    # Add coordinate markers to the updated image.
                                    if not self.config['language_only']:
                                        for i, img_path in enumerate(img_path_list):
                                            if 'front_rgb' in img_path:
                                                img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                                    if self.config['detection_box'] and not self.config['language_only']:
                                        img_path_list = annotate_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list, observation=obs, env=self.env, source=self.config.get('bbox_source', 'yolo'), object_names=implicit_name_to_real_name, numbered_names=True)
                                        if self.config['multistep']:
                                            if image_history[-1].split('.png')[0] in img_path_list[0]:
                                                image_history.pop()
                                                image_history.append(img_path_list[0])


                                print("debug [except] lmp_env.done {} current_step_in_episode {}".format(lmp_env.done, current_step_in_episode))
                                continue
                            except EpisodeDeadlineExceeded:
                                deadline.cancel()
                                flush = getattr(self, '_flush_action_diagnostics', None)
                                if flush is not None:
                                    flush()
                                attempt.update(status='timeout', failure_type='TIMEOUT')
                                lmp_env.episode_info['failure_list'].append(FailureType.TIMEOUT)
                                episode_info['termination'] = 'time_limit'
                                episode_info['timeout'] = True
                                episode_info['timeout_kind'] = 'episode_deadline'
                                done = True
                                break
                            except Exception as error:
                                flush = getattr(self, '_flush_action_diagnostics', None)
                                if flush is not None:
                                    flush()
                                attempt.update(status='error', error_type=type(error).__name__)
                                logger.exception("Generated-code episode failed")
                                raise
                            finally:
                                flush = getattr(self, '_flush_action_diagnostics', None)
                                if flush is not None:
                                    flush()
                                attempt.update(
                                    actions_executed=self.env.robot_actions_executed-actions_before,
                                    elapsed_seconds=time.monotonic()-attempt_started,
                                    failure_types=[value.name for value in
                                                   lmp_env.episode_info['failure_list'][failure_count_before:]])
                                self._record_attempt_feedback(lmps, lmp_env, feedback_count_before, attempt)
                    except EpisodeDeadlineExceeded:
                        flush = getattr(self, '_flush_action_diagnostics', None)
                        if flush is not None:
                            flush()
                        episode_info.update(termination='time_limit', timeout=True,
                                            timeout_kind='episode_deadline')
                        lmp_env.episode_info['failure_list'].append(FailureType.TIMEOUT)
                        if episode_info['program_attempts']:
                            episode_info['program_attempts'][-1].update(
                                status='timeout', failure_type='TIMEOUT')
                    except Exception as error:
                        flush = getattr(self, '_flush_action_diagnostics', None)
                        if flush is not None:
                            flush()
                        # Preserve simulator/provider faults without attributing them to the model.
                        pending_error = error
                        episode_info.update(termination='infrastructure_error',
                                            infrastructure_error=type(error).__name__)
                        from responsible_robot_bench.execution.deadline import is_model_request_timeout
                        if is_model_request_timeout(error):
                            lmp_env.episode_info['failure_list'].append(FailureType.TIMEOUT)
                            episode_info['timeout_kind'] = 'model_request_timeout'
                        lmp_env.episode_info['failure_list'].append(FailureType.UNVALID_EXP)
                    finally:
                        deadline.cancel()
                        self._active_deadline = None
                        if getattr(self, '_diagnostic_context', None) is not None:
                            self._diagnostic_context.__exit__(None, None, None)
                            self._diagnostic_context = None
                        if episode_info['program_attempts']:
                            final_attempt = episode_info['program_attempts'][-1]
                            final_attempt['failure_types'] = [value.name for value in
                                lmp_env.episode_info['failure_list'][final_attempt['failure_start_index']:]]
                            terminal_failure = ('TIMEOUT' if episode_info.get('timeout') else
                                'UNVALID_EXP' if pending_error is not None else None)
                            if terminal_failure:
                                final_attempt.update(status='timeout' if terminal_failure == 'TIMEOUT' else 'error',
                                                     failure_type=terminal_failure)
                                event = {'action': 'program attempt', 'env_feedback': terminal_failure}
                                if event not in final_attempt.setdefault('feedback', []):
                                    final_attempt['feedback'].append(event)
                                    try:
                                        for interface in lmps.values():
                                            interface.lmp_planner.update_info(copy.deepcopy(event))
                                    except Exception as error:
                                        final_attempt['feedback_error'] = type(error).__name__ + ': ' + str(error)
                    episode_info.setdefault('termination', 'environment_done' if done else 'program_attempt_limit')
                    episode_info['implementation'] = 'original_multimodule_lmp'
                    episode_info['bbox_source'] = self.config.get('bbox_source', 'yolo')
                    # done
                    # start to record the episode information

                    # lmp_env episode_info {'reward_action': [], 'reward_safety': [], 'reward_cost': [], 'reward_task': [], 'action_success': [], 'failure_list': [], 'env_feedback': [], 'episode_elapsed_seconds': [], 'action': []}
                    episode_info['instruction'] = user_instruction
                    episode_info['reward_action'] = lmp_env.episode_info['reward_action'] if len(lmp_env.episode_info['reward_action']) > 0 else [-1]
                    episode_info['reward_safety'] = lmp_env.episode_info['reward_safety'] if len(lmp_env.episode_info['reward_safety']) > 0 else [-1]
                    episode_info['reward_task'] = lmp_env.episode_info['reward_task'] if len(lmp_env.episode_info['reward_task']) > 0 else [-1]
                    episode_info['reward_cost'] = lmp_env.episode_info['reward_cost'] if len(lmp_env.episode_info['reward_cost']) > 0 else [1.0]
                    episode_info['action_success'] = lmp_env.episode_info['action_success'] if len(lmp_env.episode_info['action_success']) > 0 else [-1]

                    episode_info['failure_list'] = [error_code.value for error_code in lmp_env.episode_info['failure_list']]
                    episode_info["episode_elapsed_seconds"] = lmp_env.episode_info['episode_elapsed_seconds']       # Multiple Episode times
                    episode_info['env_feedback'] = lmp_env.episode_info['env_feedback']
                    episode_info['action'] = lmp_env.episode_info['action']
                    episode_info['controller_action_trace'] = copy.deepcopy(lmp_env.episode_info.get('controller_action_trace', []))


                    valid_rewards = [value for value in episode_info['reward_task'] if value >= 0.0]
                    if len(valid_rewards) > 0:
                        episode_info['avg_reward'] = float(np.mean(valid_rewards))
                    else:
                        episode_info['avg_reward'] = -1  # Or avg_reward = 0.0, depending on the semantic you need

                    # Read final simulator state; intermediate rewards do not define Safe/SSR.
                    try:
                        episode_info.update(terminal_labels(self.env.task._task))
                        episode_info['terminal_labels_recorded'] = True
                    except Exception as error:
                        episode_info.update(task_success=None, safety_success=None, safety_task_success=None,
                                            cost=None, terminal_labels_recorded=False)
                        episode_info.setdefault('finalization_errors', []).append({
                            'stage': 'terminal_labels', 'error_type': type(error).__name__, 'error': str(error)})
                        episode_info['failure_list'].append(FailureType.UNVALID_EXP.value)
                        if pending_error is None:
                            pending_error = error
                    episode_info['num_steps'] = voxposer_ui.lmp_planner.planner_steps              # Steps: The number of times composer calls
                    episode_info['planner_steps'] = lmp_env.execute_steps          # Planner steps: The number of times the robot executes
                    episode_info['planner_output_error'] = self.output_json_error       # Planner output error
                    # transfer the action from array to list, action may be None
                    print("RRB_MANIPULATION_EVALUATOR.py: episode_info['action'] {}".format(str(episode_info['action'])))

                    episode_info['action'] = [
                        action.tolist() if isinstance(action, np.ndarray) else
                        action if isinstance(action, str) else
                        -1 if action is None or action == -1 else
                        action  # fallback for other already-list-like objects
                        for action in episode_info['action']
                    ]


                    # Artifact or cleanup failures must not prevent the episode
                    # record or replace an earlier provider/controller exception.
                    for interface, name in ((voxposer_ui, 'planner'), (composer_ui, 'composer'),
                            (parse_query_obj_ui, 'parse_query_obj'), (get_affordance_map_ui, 'get_affordance_map'),
                            (get_avoidance_map_ui, 'get_avoidance_map'), (get_velocity_map_ui, 'get_velocity_map'),
                            (get_rotation_map_ui, 'get_rotation_map'), (get_gripper_map_ui, 'get_gripper_map')):
                        try:
                            self.save_planner_code_outputs(interface.exec_hist, name)
                            interface.clear_exec_hist()
                        except Exception as error:
                            episode_info.setdefault('finalization_errors', []).append({
                                'stage': 'planner_artifact', 'module': name,
                                'error_type': type(error).__name__, 'error': str(error)})
                            episode_info['failure_list'].append(FailureType.UNVALID_EXP.value)
                            if pending_error is None:
                                pending_error = error
                    self.save_episode_metric(episode_info)

                    progress_bar.update()
                    if pending_error is not None:
                        raise pending_error
                self.print_task_eval_results(filename="summary.json")
                self.env.close()


            # Store configuration
            with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

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
    parser.add_argument('--eval_sets', default='debug', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
    parser.add_argument('--chat_history', type=int, default=0, help='Whether to use chat history.')
    parser.add_argument('--n_shots', type=int, default=20)
    parser.add_argument('--multiview', type=int, default=0)
    parser.add_argument('--detection_box', type=int, default=1)
    parser.add_argument('--multistep', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default="baseline")
    parser.add_argument('--visual_icl', type=int, default=0)
    parser.add_argument('--cognition', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1, help='number of tensor parallel splits of the model parameters')
    parser.add_argument('--headless', type=int, default=0, help='')


    parser.add_argument('--traj_type', type=str, default="code", help='code or pre-defined or point')

    parser.add_argument('--example_classification_type', type=str, default="different_task_with_hazard", help='"task", "same_task_with_hazard", "same_task_without_hazard", "different_task_with_hazard", "different_task_without_hazard"')
    parser.add_argument('--instruction_type', type=str, default='normal', choices=['normal', 'attack', 'defense'])
    parser.add_argument('--output_dir')
    parser.add_argument('--data_root')
    parser.add_argument('--expected_episodes', type=int, default=100)
    parser.add_argument('--selected_indexes', type=lambda s: [int(i) for i in s.split(',') if i], default=[0])
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
        'instruction_type': args.instruction_type,
        'output_dir': args.output_dir,
        'data_root': args.data_root,
        'expected_episodes': args.expected_episodes,
        'max_episode_steps': args.max_episode_steps,

    }
    print("printing config ...")
    for config_key in config:
        print(f"{config_key}: {config[config_key]}")
    evaluator = RRB_ManipulationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
