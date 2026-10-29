from rrb.core import TASKS
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
from responsible_robot_bench.main import logger

VISUAL_ICL_EXAMPLES_PATH = "responsible_robot_bench/evaluator/config/visual_icl_examples/rrb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    name.split('_', 1)[0]: name for name in TASKS.values()
}

class ManipPlanner():
    def __init__(self, model_name, model_type, system_prompt, examples, n_shot=0, obs_key='front_rgb', chat_history=False, language_only=False, multiview=False, multistep=False, visual_icl=False, tp=1, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        # Instantiate LLM VLM model, input model name, model type
        self.model = RemoteModel(model_name, model_type, language_only, tp, 'manip')
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.multi_view = multiview
        self.multi_step_image = multistep
        self.visual_icl = visual_icl

    # Handle Prompt
    def process_prompt(self, user_instruction, avg_obj_coord, task_variation, prev_act_feedback=[]):


        # avg_obj_coord:  {'object 1': [73, 15, 18], 'object 2': [57, 20, 18], 'object 3': [66, 38, 18], 'object 4': [33, 43, 17], 'object 5': [57, 74, 17], 'object 6': [28, 80, 17], 'object 7': [26, 91, 17]}


        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])]))
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
        # If historical information is used.
        elif self.chat_history:
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
            task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
        else:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])]))
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        return general_prompt, task_prompt

    def process_prompt_visual_icl(self, user_instruction, avg_obj_coord, prev_act_feedback=[]):
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

    def get_message(self, images, prompt, task_prompt, messages=[]):
        if self.language_only and not self.visual_icl:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + task_prompt}],
                }
            ]
        else:
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

            return current_message

    def get_message_visual_icl(self, images, first_prompt, task_prompt, task_variation, messages=[]):
        # There is currently no system role, only a user role.
        # Why?
        # The first prompt stored in the history data.
        current_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_prompt}
                ],
            }
        ]

        visual_task_variation = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
        task_specific_image_example_path = osp.join(VISUAL_ICL_EXAMPLES_PATH, visual_task_variation)    # Load Visual ICL example
        icl_text_examples = self.examples[task_variation]                                           # Load ICL text examples
        stop_idx = 2
        for example_idx, example in enumerate(icl_text_examples):   # Iterate over text ICL examples
            if example_idx >= stop_idx:
                break
            current_image_example_path = osp.join(task_specific_image_example_path, f"episode_{example_idx+1}_step_0_front_rgb_annotated.png")
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
        # Inserted two examples in a loop
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

    def json_to_action(self, output_text):
        """Validate a pose plan without generating fallback robot motion."""
        from rrb.planning.legacy import parse_legacy_poses
        from rrb.planning.protocol import InvalidPlan
        try:
            if hasattr(self, 'sanitize_output'):
                output_text = self.sanitize_output(output_text)
            actions, response = parse_legacy_poses(output_text)
            return actions, response
        except InvalidPlan:
            self.output_json_error += 1
            return [], output_text

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def act(self, observation, user_instruction, avg_obj_coord, task_variation):
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
            full_example_prompt, task_prompt = self.process_prompt(user_instruction, avg_obj_coord, task_variation, prev_act_feedback=self.episode_act_feedback)


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
        action, json_output = self.json_to_action(out)

        # out:  {"reasoning_and_reflection":"The task is to pick up the star located at [73, 15, 18] and place it into the silver container located at [66, 38, 18]. The plan involves moving the gripper to the star's position, closing the gripper to pick it up, moving to the silver container's position, and opening the gripper to release the star into the container.",



        return action, out

    def update_info(self, info):
        # Collect user feedback information
        # Collect executed actions
        env_feedback = info['env_feedback']
        action = info['action']
        self.episode_act_feedback.append([action, env_feedback])
