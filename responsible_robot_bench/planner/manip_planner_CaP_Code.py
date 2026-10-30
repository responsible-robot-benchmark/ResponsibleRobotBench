from responsible_robot_bench.configs.task_registry import TASKS
import os
import os.path as osp
import numpy as np
import cv2
import base64
import json
import ast
import random
import time
import logging
from mimetypes import guess_type
from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import ROTATION_RESOLUTION, VOXEL_SIZE
from responsible_robot_bench.planner.remote_model import RemoteModel
from responsible_robot_bench.planner.planner_utils import local_image_to_data_url, template_manip, template_lang_manip
from responsible_robot_bench.planner.planner_utils import template_lang_manip_from_output_type, template_manip_from_output_type
from responsible_robot_bench.main import logger
from responsible_robot_bench.evaluator.evaluator_utils import FailureType, FailureException
import re
import matplotlib.pyplot as plt

from time import sleep


from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from responsible_robot_bench.envs.rrb_manipulation.amsolver.voxposer_planner.utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time

from responsible_robot_bench.envs.rrb_manipulation.amsolver.voxposer_planner.LLM_cache import DiskCache

VISUAL_ICL_EXAMPLES_PATH = "responsible_robot_bench/evaluator/config/visual_icl_examples/rrb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    name.split('_', 1)[0]: name for name in TASKS.values()
}

def composer_string_with_newlines(text):


    text = text.replace(')', ')\n')

    return text

class LMPlannerCodeInterface:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, lmp_planner, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug


        # Create system prompt
        if hasattr(lmp_planner, 'system_prompt'):

            self.system_prompt = lmp_planner.system_prompt
        else:
            self.system_prompt = "debug_prompt {variable_vars_imports}"

        # Create a prompt for example


        self._stop_tokens = list(self._cfg['stop'])                                 # End marker
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

        # Create an LMPlanner object with parameters including model name
        self.lmp_planner = lmp_planner
        self.lmp_planner.reset()    # Reset the history information inside the planner
        self.init_observations()  # Initialize observation information

    def init_observations(self):
        self.observation=None
        self.image_history=[]
        self.avg_obj_coord=None
        self.task_variation=None
        self.object_list=None
        self.available_skills=None
        self.implicit_name_to_real_name=None

    def update_observations(self, observation, image_history, avg_obj_coord, task_variation, object_list=None, available_skills=None, implicit_name_to_real_name=None):
        self.observation = observation
        self.image_history = image_history  # if use_history, it's image list, otherwise it's image_path
        self.avg_obj_coord = avg_obj_coord
        self.task_variation = task_variation
        self.object_list = object_list
        self.available_skills = available_skills
        self.implicit_name_to_real_name = implicit_name_to_real_name

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        # Import variables
        # Replace variables
        # Background knowledge information, all preloaded, do not load here
        # query_suffix
        # user_query
        # Format check
        # Return prompt and user_query
        pass

    def build_prompt_original(self, query, include_system_prompt=False):

        # This should be added before the case code, not in the system prompt

        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"    # Load variables
        else:
            variable_vars_imports_str = ''
        # Background knowledge added here
        if include_system_prompt:
            # Control whether to add system prompt here
            prompt = self.system_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)    # Replace the loaded variables in the base_prompt
        else:
            prompt = ''

        # The history added here should be code history instead of prompt history
        if self._cfg['maintain_session'] and self.exec_hist != '':  # Maintain session if needed
            prompt += f'\n{self.exec_hist}'                         # Add execution history

        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:                            # Add history information if applicable
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'


        # Prompt is a previous code generation example with current command
        # user_query is simply the user's command
        return prompt, user_query


    #     # check whether completion endpoint or chat endpoint is used
    #     if kwargs['model'] != 'gpt-3.5-turbo-instruct' and \

    #         # add special prompt for chat endpoint


    #         # handle given context (this was written originally for completion endpoint)


    #             # remove obj_context from user1

    #             # add obj_context to user2


    #             {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
    #             {"role": "user", "content": user1},
    #             {"role": "assistant", "content": assistant1},
    #             {"role": "user", "content": user2},




    #             # post processing







    def _cached_api_call(self, **kwargs):
        'Load prompt template, replace the example part, check if it exists in Cache, if not, predict'
        print("==== kwargs: {} {}".format(self._name, kwargs))

        system_prompt, task_prompt = self.lmp_planner.build_message_for_code_generation(observation=self.image_history,
                                                                                user_instruction=kwargs['prompt'],  # User's command
                                                                                avg_obj_coord=self.avg_obj_coord,
                                                                                task_variation=self.task_variation,
                                                                                object_list=self.object_list,
                                                                                available_skills=self.available_skills,
                                                                                implicit_name_to_real_name=self.implicit_name_to_real_name)

        print("==== {} system_prompt {}".format(self._name, system_prompt))
        print("=== {} task_prompt {}".format(self._name, task_prompt))


        # Contains {PROMPT_EXMAPLES}

        # Use input information kwargs to check if this result already exists
        kwargs.pop('prompt')   # Remove prompt


        kwargs_cache = {}
        # Do not reuse code cached before truncation and code-block checks.
        kwargs_cache['protocol'] = 'lmp-complete-request-v7-complete-fences'
        kwargs_cache['messages'] = json.loads(json.dumps(self.lmp_planner.episode_messages))
        kwargs_cache['user_prompt'] = task_prompt
        kwargs_cache['model'] = self.lmp_planner.model_name
        kwargs_cache['module'] = self._name
        if 'Qwen2-VL-7B-Instruct' in self.lmp_planner.model_name:
            # Cache entries must reflect the effective Code request budget,
            # not just the legacy module-local max_tokens setting.
            kwargs_cache['qwen_code_output_tokens'] = os.environ.get('RRB_CODE_MAX_OUTPUT_TOKENS', '4096')
        kwargs_cache['temperature'] = kwargs['temperature']
        kwargs_cache['max_tokens'] = kwargs['max_tokens']
        kwargs_cache['stop'] = kwargs['stop']

        # If using an image as input, it is likely different, so if using an image, there is no need to cache
        # Agent using an image:
        # Agent not using an image: planner uses description to call composer, composer (breaks down the description into known functions)
        # VLM is needed for planner. Only planner does not need to consider self.cache
        # VLM is not needed for composer
        # No need for the remaining agents, as the object already knows.


        if self._cfg['load_from_cache'] and kwargs_cache in self._cache:   # If looking in the cache.
            print('(using cache)', end=' ')
            print("A cached result already exists")


            self.lmp_planner.planner_steps += 1 # If the result is already known, planner_steps+1
            print("(using cache) planner name {} planner_steps {}".format(self.lmp_planner.planner_name, self.lmp_planner.planner_steps))
            return self._cache[kwargs_cache]
        else:
            # Input scene object, object name, input model's message

            # observation: OBS can be deleted, as the cache is already defined
            # user_instruction: The user command is in messages, can be deleted
            # avg_obj_coord: The object's position does not affect the code, can be deleted
            # task_variation：
            # object_list: Scene objects, do not affect code comparison, can be deleted
            # available_skills: If using code, this is not needed
            # implicit_name_to_real_name: TBD


            # reflection: Only used in the planner, not needed after the composer, just need to convert to code

            # Agent prediction

            self.lmp_planner.model.stop = list(kwargs['stop'])
            self.lmp_planner.model.max_tokens = kwargs['max_tokens']
            code_action, out, output_error_type = self.lmp_planner.act_code_generation() # action: parsed result, out: model output result
            if output_error_type != FailureType.NO_ERROR:
                raise FailureException(output_error_type)

            # post processing
            ret = code_action
            self._cache[kwargs_cache] = ret # Use dictionary as key to store code results
            print("planner name {} planner_steps {}".format(self.lmp_planner.planner_name, self.lmp_planner.planner_steps))

            return ret
    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt_original(query)   # Obtain system prompt and user query


        start_time = time.time()
        # Provider failures propagate after the SDK's bounded retry policy.
        # Replanning belongs to the episode runner, not an unbounded API loop.
        code_str = self._cached_api_call(
            prompt=prompt, stop=self._stop_tokens,
            temperature=self._cfg['temperature'], model=self._cfg['model'],
            max_tokens=self._cfg['max_tokens'])

        print(f'*** Agent API call took {time.time() - start_time:.2f}s ***')

        if self._cfg['include_context']:
            # build execute code
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            # build log code
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        # global variables and local variables
        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs


        if not self._name in ['composer', 'planner']:
            # if has return value, modify code format, return function
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:    # Execute code
                for s in action_str:
                    code_exec_ret_type = exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)   # Execute code, skip action line
            except Exception as e:
                print(f'Error: {e}')
                raise
        else:
            code_exec_ret_type = exec_safe(to_exec, gvars, lvars)    # Execute code snippet (define function, etc.)


        # If code error, return error type and end

        self.exec_hist += f'\n{to_log.strip()}'
        if code_exec_ret_type != FailureType.NO_ERROR:  # If no error, continue execution
            raise FailureException(code_exec_ret_type)  # If error, return error type


        # If maintain, update local variables
        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            # If calling parse_query_obj, return a DynamicObservation object, allowing access to the latest obs
            if self._name == 'parse_query_obj': # If
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            # Return value
            return lvars[self._cfg['return_val_name']]  # Return value


def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    """Run one generated program once and classify an execution failure.

    This is a compatibility executor, not a security sandbox. Generated
    programs use the robot APIs supplied by the caller.
    """
    if not isinstance(code_str, str):
        return FailureType.FORMAT_ERROR
    try:
        tree = ast.parse(code_str)
    except (SyntaxError, ValueError):
        return FailureType.FORMAT_ERROR
    if any(isinstance(node, ast.Attribute) and node.attr.startswith('__') or
           isinstance(node, ast.Name) and node.id.startswith('__')
           for node in ast.walk(tree)):
        return FailureType.UNVALID_SKILL_OR_FUNCTION

    # The original commented out the remainder of an import line. Do not
    # accidentally execute additional semicolon-separated statements there.
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    imports = [node for node in statements if isinstance(node, (ast.Import, ast.ImportFrom))]
    if any(other is not imp and other.lineno == imp.end_lineno
           and other.col_offset > imp.col_offset
           for imp in imports for other in statements):
        return FailureType.UNVALID_SKILL_OR_FUNCTION

    # The selected original executor ignored generated imports: APIs are
    # supplied by the LMP environment, not imported by generated programs.
    # Replace statements with pass so nested suites remain valid; never execute
    # imports or bind names that are absent from the supplied environment.
    class IgnoreImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.copy_location(ast.Pass(), node)

        def visit_ImportFrom(self, node):
            return ast.copy_location(ast.Pass(), node)

    tree = IgnoreImports().visit(tree)
    namespace = dict(gvars or {})
    namespace.update(lvars or {})
    namespace.update({'exec': None, 'eval': None})
    try:
        exec(compile(tree, '<lmp-program>', 'exec'), namespace, namespace)
        return FailureType.NO_ERROR
    except FailureException as error:
        return error.failure_type
    except TimeoutError:
        # EpisodeDeadlineExceeded is a TimeoutError. The episode runner owns
        # termination; an interrupted program is not a formatting failure.
        raise
    except (NameError, AttributeError):
        return FailureType.UNVALID_SKILL_OR_FUNCTION
    except Exception:
        return FailureType.FORMAT_ERROR
    finally:
        if lvars is not None:
            lvars.update({key: value for key, value in namespace.items()
                          if key not in {'__builtins__', 'exec', 'eval'}})

class LMPlanner():
    def __init__(self, model_name,
                    model_type,
                    system_prompt,
                    examples,
                    example_context,
                    n_shot=0,
                    obs_key='front_rgb',
                    chat_history=False,
                    language_only=False,
                    multiview=False,
                    multistep=False,
                    visual_icl=False,
                    tp=1,   # task type
                    traj_type="code",
                    planner_name = None,
                    text_prompt_with_semantic=True,
                    kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.example_context = example_context    # example have to be included
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        # Instantiate LLM VLM model, input model name, model type
        self.model = RemoteModel(model_name=model_name, model_type=model_type, language_only=language_only, tp=tp, task_type='manip', traj_type=traj_type)
        # Retain the real LMP module identity in request records so visual
        # examples can later be traced to their original planner or composer.
        self.model.rrb_lmp_module = planner_name
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.multi_view = multiview
        self.multi_step_image = multistep
        self.visual_icl = visual_icl

        self.traj_type = traj_type  # "code" or "point" or "pre-defined"
        assert self.traj_type in ["code", "point", "pre-defined"], "Invalid agent trajectory type {}".format(self.traj_type)
        self.text_prompt_with_semantic = text_prompt_with_semantic

        self.available_skills = None
        self.object_list = None

        self.planner_name = planner_name # default: None

        self.episode_act_feedback = []


    def process_sample_prompt(self, user_instruction):
        user_instruction = user_instruction.rstrip('.')
        task_prompt = f'The human instruction is: {user_instruction}.'
        return task_prompt


    # Handle Prompt

    # When handling prompt, choose whether to include or exclude scene information. Scene information is already known.
    def _selected_examples(self, task_variation):
        """Select countable demonstrations without repeating or inventing records."""
        if type(self.n_shot) is not int or self.n_shot < 0:
            raise ValueError('n_shot must be a nonnegative integer')
        if self.n_shot == 0:
            return []
        examples = self.examples
        if isinstance(examples, dict):
            if task_variation not in examples:
                raise ValueError(f'Requested {self.n_shot} examples for {task_variation}, but none are available')
            examples = examples[task_variation]
        if not isinstance(examples, (list, tuple)):
            raise ValueError(f'Requested {self.n_shot} examples for {task_variation}; a countable example list is required')
        if len(examples) < self.n_shot:
            raise ValueError(f'Requested {self.n_shot} examples for {task_variation}, but only {len(examples)} are available')
        selected = list(examples[:self.n_shot])
        if any(not isinstance(example, str) or not example.strip() for example in selected):
            raise ValueError(f'Missing recorded example text for {task_variation}')
        return selected

    def _example_text(self, task_variation):
        # Opaque module prompts are inherited API context, not countable records.
        if isinstance(self.examples, str):
            if type(self.n_shot) is not int or self.n_shot < 0:
                raise ValueError('n_shot must be a nonnegative integer')
            return self.examples if self.n_shot else ''
        return '\n'.join(self._selected_examples(task_variation))

    def _context_text(self):
        # Keep the original base prompt for zero-shot modules, but do not
        # send the same examples twice when both fields contain the same data.
        if self.n_shot > 0 and self.example_context == self.examples:
            return ''
        if isinstance(self.example_context, str):
            return self.example_context
        if isinstance(self.example_context, (list, tuple)) and all(
                isinstance(item, str) for item in self.example_context):
            return '\n'.join(self.example_context)
        raise ValueError('Code context must be text or a list of text blocks')

    def _visual_examples(self, task_variation):
        if getattr(self, 'require_recorded_code_visual_pairs', False):
            if self.n_shot == 0:
                return []
            try:
                category = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
            except KeyError:
                raise ValueError(f'No visual ICL image category for {task_variation}') from None
            from responsible_robot_bench.evaluator.config.rrb_manipulation_example import _code_visual_pairs
            return _code_visual_pairs(category, self.planner_name, self.n_shot)
        examples = self._selected_examples(task_variation)
        if not examples:
            return []
        try:
            category = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
        except KeyError:
            raise ValueError(f'No visual ICL image category for {task_variation}') from None
        pairs = []
        for index, example in enumerate(examples, 1):
            path = osp.join(VISUAL_ICL_EXAMPLES_PATH, category,
                            f'episode_{index}_step_0_front_rgb_annotated.png')
            if not osp.isfile(path):
                raise FileNotFoundError(f'Missing requested visual ICL image: {path}')
            pairs.append((example, path))
        return pairs

    def process_prompt(self, user_instruction, avg_obj_coords, obj_names_with_ids, available_skills, task_variation, implicit_name_to_real_name, prev_act_feedback=[]):
        """Build the task prompt from the instruction and scene."""


        # Instruction: Pick up the star and place it into the silver container.
        # user_instruction:  Pick up the star and place it into the silver container.
        # avg_obj_coord:  {'object 1': [73, 15, 18], 'object 2': [57, 20, 18], 'object 3': [66, 38, 18], 'object 4': [33, 43, 17], 'object 5': [57, 74, 17], 'object 6': [28, 80, 17], 'object 7': [26, 91, 17]}

        user_instruction = user_instruction.rstrip('.')
        # Historical action feedback information
        if len(prev_act_feedback) > 0: # If historical information is used, historical action feedback exists
            history_action_prompt = f'The human instruction is: {user_instruction}.'
            history_action_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                history_action_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
        else:
            history_action_prompt = ''

        # Trajectory type information
        if self.traj_type == "code":
            natural_traj_type = "executable code"
            # Should use the previous version, the version without reflection, directly predict the code
            # The input is n_shot_example_prompt
            n_shot_example_prompt = self._example_text(task_variation)
            context_example_prompt = self._context_text()

            input_prompt = output_prompt = ""
            # there is function to design objects according to include_context

            planner_name = self.planner_name    # Corresponds to different functional agents
        else:
            raise ValueError("Invalid trajectory type: {}".format(self.traj_type))

        # If feedback from previous actions is not used
        if len(prev_act_feedback) == 0: # No history interaction
            if self.traj_type == "code":
                # Code generation: No need for space VOXL_SIZE and other information in system_prompt

                # If a case is added to the system prompt

                # Already switched in


                general_prompt = self.system_prompt.format(PROMPT_EXAMPLES="")  # No need for case information


                # Adding cases to the system prompt does not stabilize the predicted code.
                # The user_instruction contains instruction information.
                if self.n_shot >= 1:
                    task_prompt = f"\n{context_example_prompt}\n{n_shot_example_prompt}\n{user_instruction}"
                else:
                    task_prompt = f"\n{context_example_prompt}\n{user_instruction}"
                print("==== n_shot_example_prompt: ", n_shot_example_prompt)
                print("==== user_instruction: ", user_instruction)


            else:
                raise ValueError("Invalid trajectory type: {}".format(self.traj_type))

        # If historical information is used.
        # Distinguish between historical CHAT information and historical action feedback.
        elif self.chat_history: # There is historical feedback (prev_act_feedback is not empty), but self.chat_history is True.
            # If historical CHAT messages are used, the prompt needs historical action information, and the message needs historical chat information.
            # If Chat History is used, it indicates that the first step has been executed. The history chat information will be loaded later.
            general_prompt = history_action_prompt # Use historical messages as background information.

            if self.traj_type == "point":
                task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
            elif self.traj_type == "code":
                if self.n_shot >= 1:

                    task_prompt = f'''\n\n ##Considering the above interaction history and the current image state, to achieve the follow human instruction.\n{n_shot_example_prompt}\n{user_instruction}'''

                else:

                    task_prompt = f'''\n\n ##Considering the above interaction history and the current image state, to achieve the follow human instruction.\n{user_instruction}'''

            elif self.traj_type == "pre-defined":
                task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable code with pre-defined skills.'''

        else:   # Historical action feedback exists, but self.chat_history is False
            # Historical action feedback exists, indicating the first action has been executed
            # Historical action feedback exists, so it needs to be added
            if self.traj_type == "code":


                general_prompt = self.system_prompt.format(PROMPT_EXAMPLES='')  # No need for case information
                if self.n_shot >= 1:
                    task_prompt = f"\n## Below are some examples to guide you in completing the task.\n{context_example_prompt}\n{n_shot_example_prompt}\n ## Now you are supposed to follow the above examples to generate {natural_traj_type} that completes the below human instruction. \n{user_instruction}."
                else:
                    task_prompt = f"\n## Context examples:\n{context_example_prompt}\n ## Now you are supposed to follow the above examples to generate {natural_traj_type} that completes the below human instruction. \n{user_instruction}."
            else:
                raise ValueError("Invalid trajectory type: {}".format(self.traj_type))


            # If self.chat_history is False, prompt still needs to include historical action feedback, but self.messages construction process does not include historical chat information.
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        print("== general_prompt: ", general_prompt)
        print("== task_prompt: ", task_prompt)

        # Return background information and task information
        return general_prompt, task_prompt


    def process_prompt_visual_icl(self, user_instruction, avg_obj_coords, obj_names_with_ids, available_skills, task_variation, implicit_name_to_real_name, prev_act_feedback=[]):
        """Build the task prompt from the instruction and scene."""


        # Instruction: Pick up the star and place it into the silver container.
        # user_instruction:  Pick up the star and place it into the silver container.
        # avg_obj_coord:  {'object 1': [73, 15, 18], 'object 2': [57, 20, 18], 'object 3': [66, 38, 18], 'object 4': [33, 43, 17], 'object 5': [57, 74, 17], 'object 6': [28, 80, 17], 'object 7': [26, 91, 17]}

        user_instruction = user_instruction.rstrip('.')
        # Historical action feedback information
        if len(prev_act_feedback) > 0: # If historical information is used, historical action feedback exists
            history_action_prompt = f'The human instruction is: {user_instruction}.'
            history_action_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                history_action_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
        else:
            history_action_prompt = ''

        # Trajectory type information
        if self.traj_type == "code":
            natural_traj_type = "executable code"

            if implicit_name_to_real_name == None:


                input_prompt = ""
            else:
                if "objects = " in user_instruction:
                    user_instruction = re.sub(r'((?:\s*\n)+)(?= *string)', '', user_instruction)
                    input_prompt = "" + "\n# object_names_for_the_objects_in_image: {}".format(implicit_name_to_real_name) # Current scene object description information, as input
                else:
                    input_prompt = ""


            # self.examples needs the corresponding image

            context_example_prompt = self._context_text()
        else:
            raise ValueError("Invalid trajectory type: {}".format(self.traj_type))
        # If feedback from previous actions is not used
        if len(prev_act_feedback) == 0: # No history interaction

            if self.traj_type == "code":
                # Code generation: No need for space VOXL_SIZE and other information in system_prompt


                # Code generation requires necessary context information.


                general_prompt = self.system_prompt.format(PROMPT_EXAMPLES='')  # No need for case information
                task_prompt = f"\n{context_example_prompt}\n{input_prompt}\n{user_instruction}"


            else:
                raise ValueError("Invalid trajectory type: {}".format(self.traj_type))
        # If historical information is used.
        # Distinguish between historical CHAT information and historical action feedback.
        elif self.chat_history: # There is historical feedback (prev_act_feedback is not empty), but self.chat_history is True.
            # If historical CHAT messages are used, the prompt needs historical action information, and the message needs historical chat information.
            # If Chat History is used, it indicates that the first step has been executed. The history chat information will be loaded later.
            general_prompt = history_action_prompt # Use historical messages as background information.

            # The task prompt for Visual ICL does not differ.
            if self.traj_type == "code":

                task_prompt = f'''\n\n ##Considering the above interaction history and the current image state, to achieve the follow human instruction.\n{context_example_prompt}\n\n{input_prompt}\n{user_instruction}'''

            else:
                raise ValueError("Invalid trajectory type: {}".format(self.traj_type))
        else:   # Historical action feedback exists, but self.chat_history is False
            # Historical action feedback exists, indicating the first action has been executed
            # Historical action feedback exists, so it needs to be added
            if self.traj_type == "code":
                general_prompt = self.system_prompt.format(PROMPT_EXAMPLES='')
            else:
                raise ValueError("Invalid trajectory type: {}".format(self.traj_type))


            task_prompt = f"## Context Examples:\n\n{context_example_prompt}\n ## Now you are supposed to follow the above examples to generate {natural_traj_type} that completes the below human instruction. \n\n{input_prompt}\n{user_instruction}"


            # Historical action feedback?
            # If self.chat_history is False, prompt still needs to include historical action feedback, but self.messages construction process does not include historical chat information.
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        # Return background information and task information
        return general_prompt, task_prompt


    def process_prompt_visual_icl_original(self, user_instruction, avg_obj_coord, prev_act_feedback=[]):
        # If using visual icl
        """Plan the next action from the task, object coordinates, and feedback.

        user_instruction is the user task; avg_obj_coord describes object positions.
        prev_act_feedback contains (action, environment feedback) pairs. With chat
        history enabled, inspect the current image, describe progress, and use prior
        failures when revising the plan. Otherwise, condition on recorded feedback."""
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0: # No history interaction
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
        elif self.chat_history: # There is historical interaction (prev_act_feedback is not empty), but self.chat_history is True
            # general_prompt directly describes the task
            # Then concatenate the history interaction:
            # task_prompt becomes more complex, requiring:
                # Parse the current visual state (get information from the image).
                # Summarize the interaction history and environmental feedback.
                # Analyze why the previous action failed (if the task is not completed).
                # Generate a new action plan.
                # Output a new 7-dimensional executable grasp action.
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
            task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
        else:   # There is a history interaction, but self.chat_history is False
            # In this case, similar to case 1, but with additional prev_act_feedback history.
            # Here, there is no explicit requirement to summarize the visual state, but rather to adjust the strategy based solely on existing actions and feedback.
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        return general_prompt, task_prompt


    def get_message_cap(self, instruct:str, task_prompt:str, images, messages=[]):
        messages = messages or [
                    {
                        "role": "system",
                        "content": [
                        {
                            "type": "text",
                            "text": instruct
                        }
                        ]
                    }
                    ]

        # # History is empty
        # Create input message
        # get role
        role = "user"
        # get content
        content = []
        if self.language_only and not self.visual_icl:
            content.append({"type": "text", "text": task_prompt})
        else:
            if self.multi_step_image:
                # use the last three imags as multi-step images
                pass
            else:
                #
                pass

        # if not empty, first role shoule be system
        if len(messages) > 0:
            assert messages[0]["role"] == "system"
            return messages + [
                {
                    "role": role,
                    "content": content
                }
            ]
        else:
            return messages + [
                {
                    "role": role,
                    "content": content
                }
            ]

    def get_message_general(self, images, instruct: str, task_prompt: str, messages=[]):
        if len(messages) == 0:
            system_messages = [
                {
                    "role": "system",
                    "content": [
                    {
                        "type": "text",
                        "text": instruct
                    }
                    ]
                }
                ]
        else:
            system_messages = messages[0:1]

        # If only language is used, without visual understanding
        content = []
        if self.language_only and not self.visual_icl:
            # No need to load visual information, nor historical cases
            # No need for image information
            if len(messages) == 0:
                role = "user"
                content.append({"type": "text", "text": task_prompt})   # If the first input prompt
            else:
                role = "user"
                content.append({"type": "text", "text": instruct + task_prompt})

            #     {


            #             {"type": "text", "text": prompt + task_prompt}],


            return system_messages + [{
                "role": role,
                "content": content
            }]
        else:

            # If visual information is needed
            if self.multi_step_image:

                #     {


                #             # {"type": "text", "text": prompt}



                # system_messages

                # use the last three imags as multi-step images
                if len(images) >= 3:
                    multi_step_images = images[-3:-1]
                    # add the current task prompt and input image
                    current_message = [
                        {
                            "role": "user",
                            "content": [
                                # {"type": "text", "text": prompt}
                                ],
                        }
                    ]
                    # If it contains historical information, the current instruction is the content containing historical information, which needs to be placed together with the task_prompt
                    if len(messages) > 0:
                        current_message[0]["content"].append(
                            {
                                "type": "text",
                                "text": instruct,
                            }
                        )
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": "You are given the scene observations from the last two steps:",
                        }
                    )
                    for image in multi_step_images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)
                        data_url = local_image_to_data_url(image_path=image_path)
                        # Add image information
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )


                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )


                    # add the current step image
                    current_step_image = images[-1]
                    if type(current_step_image) == str:
                        image_path = current_step_image
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, current_step_image)
                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
                    current_message = system_messages + current_message
                else:
                    # If there is only one photo

                    current_message = [
                        {
                            "role": "user",
                            "content": [
                                # {"type": "text", "text": full_prompt}
                                ],
                        }
                    ]
                    if len(messages) > 0:   # If it contains historical information, the current instruction is the content containing historical information, which needs to be placed together with the task_prompt
                        current_message[0]["content"].append(
                            {
                                "type": "text",
                                "text": instruct,
                            }
                        )
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )

                    for image in images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)

                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )
                    current_message = system_messages + current_message

            else:   # If it is not necessary to use multiple-step images

                # If multiple perspectives are used, the images contain multiple perspectives of images


                current_message = [
                    {
                        "role": "user",
                        "content": [
                            # {"type": "text", "text": full_prompt}
                            ],
                    }
                ]
                if len(messages) > 0:   # If it contains historical information, the current instruction is the content containing historical information, which needs to be placed together with the task_prompt
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": instruct,
                        }
                    )
                current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )
                for image in images:    # If there are multiple perspectives images, put the data of multiple images in, otherwise, put one image
                    if type(image) == str:
                        image_path = image
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, image)

                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
                current_message = system_messages + current_message


            return current_message


    def get_message(self, images, prompt, task_prompt, messages=[], debug_vis=True):
        # If only language is used, not using visual contrast understanding
        if self.language_only and not self.visual_icl:
            # No need to load visual information, nor historical cases
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + task_prompt}],
                }
            ]
        else:
            # If visual information is needed
            if self.multi_step_image:
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}],
                    }
                ]

                # use the last three imags as multi-step images
                if len(images) >= 3:
                    multi_step_images = images[-3:-1]
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": "You are given the scene observations from the last two steps:",
                        }
                    )
                    for image in multi_step_images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)
                        data_url = local_image_to_data_url(image_path=image_path)
                        # Add image information
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )


                    # add the current task prompt and input image
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )

                    # add the current step image
                    current_step_image = images[-1]
                    if type(current_step_image) == str:
                        image_path = current_step_image
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, current_step_image)
                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
                else:
                    full_prompt = prompt + task_prompt
                    current_message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt}],
                        }
                    ]

                    for image in images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)

                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )

            else:
                full_prompt = prompt + task_prompt
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt}],
                    }
                ]

                for image in images:
                    if type(image) == str:
                        image_path = image
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, image)

                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
            # if debug_vis: read with BGR
            if debug_vis:
                plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1])
                plt.show()

            return current_message


    def get_message_visual_icl(self, images, first_prompt, task_prompt, task_variation, messages=[]):
        if len(messages) == 0:
            system_messages = [
                {
                    "role": "system",
                    "content": [
                    {
                        "type": "text",
                        "text": first_prompt
                    }
                    ]
                }
                ]
        else:
            system_messages = messages[0:1]

        if len(messages) > 0:
            current_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": first_prompt}
                    ],
                }
            ]
        else:
            current_message = [
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": first_prompt}
                    ],
                }
            ]

        for example_idx, (example, image_path) in enumerate(self._visual_examples(task_variation), 1):
            current_message[0]["content"].append({
                "type": "text", "text": "Example {}:\n{}".format(example_idx, example)})
            current_message[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": local_image_to_data_url(image_path=image_path)}})

        # The requested examples precede the current task.
        # Now add task prompt (current task)
        # add the task prompt
        current_message[0]["content"].append(
            {
                "type": "text",
                "text": task_prompt,    # The current task prompt includes example text.
            }
        )
        # Add the current image.
        for image in images:
            if type(image) == str:
                image_path = image
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            data_url = local_image_to_data_url(image_path=image_path)
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    }
                }
            )
        return system_messages + current_message


    def get_message_visual_icl_without_system_prompt(self, images, first_prompt, task_prompt, task_variation, messages=[]):

        # The first prompt stored in the history data.
        current_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_prompt}
                ],
            }
        ]

        for example_idx, (example, current_image_example_path) in enumerate(
                self._visual_examples(task_variation)):
            example = "Example {}:\n{}".format(example_idx+1, example)
            data_url = local_image_to_data_url(image_path=current_image_example_path)   # Load image

            # Add the example image and the corresponding text to the message
            current_message[0]["content"].append(
                {
                    "type": "text",
                    "text": example,
                }
            )
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,    # Image URL
                    }
                }
            )
        # The requested examples precede the current task.
        # Now add task prompt (current task)
        # add the task prompt
        current_message[0]["content"].append(
            {
                "type": "text",
                "text": task_prompt,
            }
        )

        for image in images:
            if type(image) == str:
                image_path = image
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            data_url = local_image_to_data_url(image_path=image_path)
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    }
                }
            )
        return current_message


    def is_action_in_list(self, a, b):
        # Remove possible quotes around 'a'.
        a_clean = re.sub(r"^['\"]|['\"]$", "", str(a))

        # Process each element in 'b', removing possible quotes.
        b_clean = [re.sub(r"^['\"]|['\"]$", "", item) for item in b]
        return a_clean in b_clean


    def json_to_action_pre_defined_skills(self, output_text):
        # Convert to action format.
        try:
            json_object = json.loads(output_text)
            action = []
            try:
                executable_plan = json_object['executable_code'] if 'executable_code' in json_object else json_object["properties"]["executable_code"]

            except:
                print("Failed to get executable plan from json object")
                print('random action')
                self.output_json_error += 1

                # Return [random_high_level_action], None   # Return random action
                return [], output_text, FailureType.FORMAT_ERROR
            if type(executable_plan) == str:    # If it's a string, it needs to be parsed as a list.
                try:
                    # Convert to executable robot actions
                    # The ast.literal_eval() function is used to safely parse and evaluate literal expressions (such as dict, list, tuple, int, float, bool, None, etc.) in a string.
                    executable_plan = ast.literal_eval(executable_plan) # Convert string to list
                except Exception as e:
                    print("Failed to decode string executable plan to list executable plan:", e)
                    print('random action')

                    self.output_json_error += 1

                    # Return [random_high_level_action], None   # Return random action
                    return [], output_text, FailureType.FORMAT_ERROR
            if type(executable_plan) == list:
                if len(executable_plan) == 1 and isinstance(executable_plan[0], str):
                    try:
                        parsed_plan = ast.literal_eval(executable_plan[0])
                        literal_eval_result = True
                    except:
                        literal_eval_result = False

                if len(executable_plan) == 1 and isinstance(executable_plan[0], str) and literal_eval_result==True:
                    try:
                        parsed_plan = ast.literal_eval(executable_plan[0])
                        print("parsed_plan after ast.literal_eval(executable_plan[0]):", parsed_plan)
                        if isinstance(parsed_plan, list):
                            executable_plan = parsed_plan
                    except:
                        print("Failed to decode string executable plan to list executable plan:", executable_plan)
                        self.output_json_error += 1
                        return [], output_text, FailureType.FORMAT_ERROR
                else:   # If elements in the list cannot be further parsed, return directly
                    executable_plan = executable_plan

            if len(executable_plan) > 0:    # Parses into action sequence
                for x in executable_plan:
                    # check whether all components are in available_skills


                    if self.is_action_in_list(x, self.available_skills) == True:
                        selected_action = x

                    else:
                        print("Failed to decode string action to list action:", e)
                        self.output_json_error += 1


                        return [], output_text, FailureType.UNVALID_OBJECT
                    action.append(selected_action)
                # Return action sequence
                return action, json_object, FailureType.NO_ERROR
            else:
                print("Empty executable plan, quit the episode ...")

                self.output_format_error += 1   # If empty, also considered parsing error
                return [], output_text, FailureType.FORMAT_ERROR

        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print('random action')

            self.output_json_error += 1


            return [], output_text, FailureType.FORMAT_ERROR

        except Exception as e:
            print("An expected error occurred:", e)
            print('random action')

            self.output_json_error += 1


            return [], output_text, FailureType.FORMAT_ERROR


    def json_to_action_code(self, output_text):
        """Return a consistent result without synthesizing fallback actions."""
        from responsible_robot_bench.planner.response_parsing import parse_code_response
        from responsible_robot_bench.planner.response_parsing import InvalidPlan
        try:
            if hasattr(self, 'sanitize_output'):
                output_text = self.sanitize_output(output_text)
            actions, response = parse_code_response(output_text)
            return actions, response, FailureType.NO_ERROR
        except InvalidPlan:
            self.output_json_error += 1
            return [], output_text, FailureType.FORMAT_ERROR

    def json_to_action(self, output_text):
        """Validate a pose plan without generating fallback robot motion."""
        from responsible_robot_bench.planner.response_parsing import parse_pose_response
        from responsible_robot_bench.planner.response_parsing import InvalidPlan
        try:
            if hasattr(self, 'sanitize_output'):
                output_text = self.sanitize_output(output_text)
            actions, response = parse_pose_response(output_text)
            return actions, response, FailureType.NO_ERROR
        except InvalidPlan:
            self.output_json_error += 1
            return [], output_text, FailureType.FORMAT_ERROR


    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0


    def act_from_pre_defined_skills(self, observation, user_instruction, avg_obj_coord, task_variation, object_list=None, available_skills=None, instruction_current_task_staus=None):
        # If using a predefined action sequence, need to add API information that can be used.
        # Handle self.system_prompt
        # If not using visual information, or using pre-defined trajectory
        if self.traj_type == "pre-defined" and object_list is not None and available_skills is not None:
            #
            self.available_skills = available_skills
            self.object_list = object_list


            if instruction_current_task_staus is not None:

                instruction_current_task_staus=f'The following objects are in scene:{object_list}.\nThe available pre-defined skills are {available_skills}'


            user_instruction=f'{user_instruction}\n{instruction_current_task_staus}'
            self.episode_messages = self.get_message_cap(images=[], instruct=self.system_prompt, task_prompt=user_instruction, messages=[])
        else:
            # Test without visual information
            self.episode_messages = self.get_message_cap(images=[], instruct=self.system_prompt, task_prompt=user_instruction, messages=[])
        for entry in self.episode_messages:
            for content_item in entry["content"]:   # content's content
                if content_item["type"] == "text":  # If text
                    text_content = content_item["text"] # Get text content, print text content
                    logger.debug(f"Model Input:\n{text_content}\n")
        # Use LLM VLM prediction results
        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
            except:
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            # If not using gemini model, directly call respond
            try:
                out = self.model.respond(self.episode_messages)
                print("===")
                print(out)
            except:
                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        for stop in getattr(self.model, 'stop', []) or []:
            if stop:
                out = out.split(stop, 1)[0]

        if self.chat_history:
            # If using historical data, store in Episode Message
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        # Utilize VLM output results with json to action processing, output actions
        #
        if self.traj_type == "pre-defined":
            action, json_output, ret_type = self.json_to_action_pre_defined_skills(out)
        else:
            action, json_output, output_error_type = self.json_to_action(out)

        # out:  {"reasoning_and_reflection":"The task is to pick up the star located at [73, 15, 18] and place it into the silver container located at [66, 38, 18]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the silver container's position, and opening the gripper to release the star into the container.",



        return action, out


    def _first_request_with_shared_feedback(self, general_prompt, task_prompt):
        """Keep system instructions when another module supplied prior feedback."""
        if self.chat_history and self.episode_act_feedback and not self.episode_messages:
            return (self.system_prompt.format(PROMPT_EXAMPLES=''),
                    general_prompt + '\n' + task_prompt)
        return general_prompt, task_prompt

    def build_message_for_code_generation(self, observation, user_instruction, avg_obj_coord, task_variation, object_list=None, available_skills=None, implicit_name_to_real_name=None):
        'Should input processed input'
        # Input image and user description, average object voxel coordinates, output action sequence
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        # Current scene state information
        self.available_skills = available_skills
        self.object_list = object_list

        if self.visual_icl and not self.language_only:  # If using visual history correction, not only use language information (use language + visual)
            # Input user description
            # Prediction
            if self.text_prompt_with_semantic == 1 or self.text_prompt_with_semantic == True:


                assert implicit_name_to_real_name is not None, "implicit_name_to_real_name should not be None when using visual ICL"
            else:
                implicit_name_to_real_name = None   # If using visual information, no need to know the object name


            full_example_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction=user_instruction,
                                                                       avg_obj_coords=avg_obj_coord,
                                                                       task_variation=task_variation,
                                                                       prev_act_feedback=self.episode_act_feedback,
                                                                       obj_names_with_ids=object_list,
                                                                       available_skills=available_skills,
                                                                       implicit_name_to_real_name=implicit_name_to_real_name)  # If using point, available_skills is None
            full_example_prompt, task_prompt = self._first_request_with_shared_feedback(
                full_example_prompt, task_prompt)
            # Print when filtering out the prompt content of the image, because it's too long

            pass  # Removed an interactive debug checkpoint.

            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                # If only language information: task_prompt + template_lang_manip
                # If using language + visual information, task_prompt + template_manip (visual information is definitely used here)
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip


            # Input OBS and description, output LLM/VLM input content
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, full_example_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, full_example_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, full_example_prompt, task_prompt, task_variation)


        else:   # If only using language information or VLM without visual ICL

            # # Visualize image


            if self.text_prompt_with_semantic == 1 or self.text_prompt_with_semantic == True:
                assert implicit_name_to_real_name is not None, "implicit_name_to_real_name should not be None when using visual ICL"
            else:
                implicit_name_to_real_name = None

            full_example_prompt, task_prompt = self.process_prompt(user_instruction=user_instruction,
                                                                avg_obj_coords=avg_obj_coord,
                                                                task_variation=task_variation,
                                                                prev_act_feedback=self.episode_act_feedback,
                                                                obj_names_with_ids=object_list,
                                                                available_skills=available_skills,
                                                                implicit_name_to_real_name=implicit_name_to_real_name)  # If using point, available_skills is None
            full_example_prompt, task_prompt = self._first_request_with_shared_feedback(
                full_example_prompt, task_prompt)


            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                # If only using language: task_prompt + template_lang_manip
                # If using language, but not using visual ICL: task_prompt + template_manip

                if self.language_only:
                    if self.traj_type == "pre-defined":
                        task_prompt = task_prompt + template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="A list of pre-defined skills", OUPUT_DESCRIPTION2="each pre-defined skill based on provided available skills")
                    elif self.traj_type == "code":

                        # If code, the output result is not in JSON format.
                        task_prompt = task_prompt #+ template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="Code", OUPUT_DESCRIPTION2="each code being a sequence of actions")
                    elif self.traj_type == "point":
                        task_prompt = task_prompt + template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_plan", OUPUT_DESCRIPTION1="A list of discrete actions", OUPUT_DESCRIPTION2="each discrete action being a 7-dimensional discrete action")
                    else:
                        raise ValueError("Invalid trajectory type")
                else:      # If using visual information
                    if self.traj_type == "pre-defined":
                        task_prompt = task_prompt + template_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="A list of pre-defined skills", OUPUT_DESCRIPTION2="each pre-defined skill based on provided available skills")
                    elif self.traj_type == "code":

                        task_prompt = task_prompt   # If outputting code, the output is not in JSON format
                    elif self.traj_type == "point":
                        task_prompt = task_prompt + template_manip_from_output_type.format(EXECUTABLE_KEY="executable_plan", OUPUT_DESCRIPTION1="A list of discrete actions", OUPUT_DESCRIPTION2="each discrete action being a 7-dimensional discrete action")
                    else:
                        raise ValueError("Invalid trajectory type")
            if len(self.episode_messages) == 0:

                self.episode_messages = self.get_message_general(obs, full_example_prompt, task_prompt)
            else:
                if self.chat_history:   # If using historical data

                    self.episode_messages = self.get_message_general(obs, full_example_prompt, task_prompt, self.episode_messages)
                else:

                    self.episode_messages = self.get_message_general(obs, full_example_prompt, task_prompt)


        print("full_example_prompt: ", full_example_prompt)
        print("task_prompt: ", task_prompt)

        return full_example_prompt, task_prompt


    def act_code_generation(self):
        # RemoteModel/its SDK owns bounded transport retries.
        out = self.model.respond(self.episode_messages)

        if self.chat_history:
            # If using historical data, store in Episode Message
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        logger.debug(f"Model Output:\n{out}\n")
        # Utilize VLM output results with json to action processing, output actions

        #     action, json_output, output_error_type = self.json_to_action_pre_defined_skills(out)

        #     action, json_output, output_error_type = self.json_to_action(out)

        self.planner_steps += 1
        if self.traj_type == "code":

            # Get the code
            # A Query marker is also a valid Python comment inside generated
            # code. Do not truncate there. If the model emits fenced code,
            # execute the first complete block rather than subsequent prose.
            fenced = re.search(r'```(?:python)?[ \t]*\n(.*?)```', out,
                               flags=re.IGNORECASE | re.DOTALL)
            incomplete_fence = fenced is None and re.search(r'(?m)^[ \t]*```', out) is not None
            action = json_output = (fenced.group(1) if fenced else out).strip("`\n ")


            # action, json_output = self.json_to_action_code(out) # Do not parse first
            metadata = getattr(self.model, 'last_response_metadata', None)
            truncated = isinstance(metadata, dict) and metadata.get('finish_reason') == 'length'
            if truncated or incomplete_fence or not action.strip():  # Reject incomplete programs.
                output_error_type = FailureType.FORMAT_ERROR
            else:
                try:
                    tree = ast.parse(action)
                    output_error_type = FailureType.NO_ERROR    # This does not execute the code, just parses. Does not test if the function exists, if the variables are reasonable.
                except:
                    print("Failed to parse the code")
                    output_error_type = FailureType.FORMAT_ERROR

        else:
            raise ValueError("Invalid trajectory type")

        print("[agent estimation result] \n {}".format(action))

        return action, out, output_error_type


    def act_general(self, observation, user_instruction, avg_obj_coord, task_variation, object_list=None, available_skills=None, implicit_name_to_real_name=None):
        # Input image and user description, average object voxel coordinates, output action sequence
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        # Current scene state information
        self.available_skills = available_skills
        self.object_list = object_list

        if self.visual_icl and not self.language_only:  # If using visual history correction, not only use language information (use language + visual)
            # Input user description
            # Prediction
            implicit_name_to_real_name = None   # If using visual information, no need to know the object name

            first_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction, avg_obj_coord, prev_act_feedback=self.episode_act_feedback)


            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                # If only language information: task_prompt + template_lang_manip
                # If using language + visual information, task_prompt + template_manip (visual information is definitely used here)
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip


            # Input OBS and description, output LLM/VLM input content
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)


        else:   # If only language information is used
            # If visual data is not used
            print("implicit_name_to_real_name: ", implicit_name_to_real_name)
            full_example_prompt, task_prompt = self.process_prompt(user_instruction=user_instruction,
                                                                avg_obj_coords=avg_obj_coord,
                                                                task_variation=task_variation,
                                                                prev_act_feedback=self.episode_act_feedback,
                                                                obj_names_with_ids=object_list,
                                                                available_skills=available_skills,
                                                                implicit_name_to_real_name=implicit_name_to_real_name)  # If using point, available_skills is None


            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                # If only using language: task_prompt + template_lang_manip
                # If using language, but not using visual ICL: task_prompt + template_manip

                if self.language_only:
                    if self.traj_type == "pre-defined":
                        task_prompt = task_prompt + template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="A list of pre-defined skills", OUPUT_DESCRIPTION2="each pre-defined skill based on provided available skills")
                    elif self.traj_type == "code":
                        task_prompt = task_prompt + template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="Code", OUPUT_DESCRIPTION2="each code being a sequence of actions")
                    elif self.traj_type == "point":
                        task_prompt = task_prompt + template_lang_manip_from_output_type.format(EXECUTABLE_KEY="executable_plan", OUPUT_DESCRIPTION1="A list of discrete actions", OUPUT_DESCRIPTION2="each discrete action being a 7-dimensional discrete action")
                    else:
                        raise ValueError("Invalid trajectory type")
                else:      # If using visual information
                    if self.traj_type == "pre-defined":
                        task_prompt = task_prompt + template_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="A list of pre-defined skills", OUPUT_DESCRIPTION2="each pre-defined skill based on provided available skills")
                    elif self.traj_type == "code":
                        task_prompt = task_prompt + template_manip_from_output_type.format(EXECUTABLE_KEY="executable_code", OUPUT_DESCRIPTION1="Code", OUPUT_DESCRIPTION2="each code being a sequence of actions")
                    elif self.traj_type == "point":
                        task_prompt = task_prompt + template_manip_from_output_type.format(EXECUTABLE_KEY="executable_plan", OUPUT_DESCRIPTION1="A list of discrete actions", OUPUT_DESCRIPTION2="each discrete action being a 7-dimensional discrete action")
                    else:
                        raise ValueError("Invalid trajectory type")


            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)
            else:
                if self.chat_history:   # If using historical data
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, self.episode_messages)
                else:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)


        # Task planning has not been performed, directly performed keypoint planning
        # Planner Output Action: [[73, 15, 18, 0, 0, 0, 1], [73, 15, 18, 0, 0, 0, 0], [66, 38, 18, 0, 0, 0, 0], [66, 38, 18, 0, 0, 0, 1]]
        # evaluator rrb_manipulation_evaluator: action_single: [73, 15, 18, 0, 0, 0, 1]


        # for content_item in entry["content"]:   # Content content
        # if content_item["type"] == "text":  # If it's text


        # Use LLM VLM prediction results
        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
            except:
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            # If not using gemini model, directly call respond
            try:
                out = self.model.respond(self.episode_messages)
            except:
                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        for stop in getattr(self.model, 'stop', []) or []:
            if stop:
                out = out.split(stop, 1)[0]

        if self.chat_history:
            # If using historical data, store in Episode Message
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        # Utilize VLM output results with json to action processing, output actions
        if self.traj_type == "pre-defined":
            action, json_output, output_error_type = self.json_to_action_pre_defined_skills(out)
        elif self.traj_type == "point":
            action, json_output, output_error_type = self.json_to_action(out)
        elif self.traj_type == "code":
            action, json_output, output_error_type = self.json_to_action_code(out)
        else:
            raise ValueError("Invalid trajectory type")

        # out:  {"reasoning_and_reflection":"The task is to pick up the star located at [73, 15, 18] and place it into the silver container located at [66, 38, 18]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the silver container's position, and opening the gripper to release the star into the container.",



        return action, out, output_error_type


    def act(self, observation, user_instruction, avg_obj_coord, task_variation):
        'Supports L with images/only language without images\nSupports with historical interaction data/without historical interaction data\nSupports'
        # Input image and user description, average object voxel coordinates, output action sequence

        # Load image
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        # If visual history correction is used, not only language information (uses language + vision)
        if self.visual_icl and not self.language_only:  #
            # Input user description
            # Prediction
            first_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction, avg_obj_coord, prev_act_feedback=self.episode_act_feedback)


            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                # If only language information: task_prompt + template_lang_manip
                # If using language + visual information, task_prompt + template_manip (visual information is definitely used here)
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip


            # Input OBS and description, output LLM/VLM input content
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)


        # If only language information is used
        else:
            # If visual data is not used
            full_example_prompt, task_prompt = self.process_prompt(user_instruction, avg_obj_coords, obj_names, obj_ids, task_variation, prev_act_feedback=self.episode_act_feedback)
            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, self.episode_messages)
                else:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)


        # Task planning has not been performed, directly performed keypoint planning
        # Planner Output Action: [[73, 15, 18, 0, 0, 0, 1], [73, 15, 18, 0, 0, 0, 0], [66, 38, 18, 0, 0, 0, 0], [66, 38, 18, 0, 0, 0, 1]]
        # evaluator rrb_manipulation_evaluator: action_single: [73, 15, 18, 0, 0, 0, 1]
        # ===============================Finish Building Episode Messages======================================================================
        for entry in self.episode_messages:
            for content_item in entry["content"]:   # content's content
                if content_item["type"] == "text":  # If text
                    text_content = content_item["text"] # Get text content, print text content
                    logger.debug(f"Model Input:\n{text_content}\n")

        # Use LLM VLM prediction results
        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
            except:
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            # If not using gemini model, directly call respond
            try:
                out = self.model.respond(self.episode_messages)
            except:
                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        for stop in getattr(self.model, 'stop', []) or []:
            if stop:
                out = out.split(stop, 1)[0]

        if self.chat_history:
            # If using historical data, store in Episode Message
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        # Utilize VLM output results with json to action processing, output actions
        action, json_output, output_error_type = self.json_to_action(out)

        # out:  {"reasoning_and_reflection":"The task is to pick up the star located at [73, 15, 18] and place it into the silver container located at [66, 38, 18]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the silver container's position, and opening the gripper to release the star into the container.",



        return action, out


    def update_info(self, info):
        # Collect user feedback information
        # Collect executed actions
        env_feedback = info['env_feedback']
        action = info['action']
        self.episode_act_feedback.append([action, env_feedback])
