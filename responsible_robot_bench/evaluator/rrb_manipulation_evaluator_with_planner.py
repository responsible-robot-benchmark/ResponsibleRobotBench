from rrb.legacy import terminal_labels
import re
import os
import numpy as np
from tqdm import tqdm
import json
import copy
import argparse

from responsible_robot_bench.evaluator.config.system_prompts import rrb_manipulation_system_prompt # traj_type: point


from responsible_robot_bench.envs.rrb_manipulation.RRBManEnv_safety import RRBManEnv, EVAL_SETS, ValidEvalSets
from responsible_robot_bench.envs.rrb_manipulation.rrb_man_utils import form_object_coord_for_input, draw_bounding_boxes, draw_xyz_coordinate

from responsible_robot_bench.planner.manip_planner_CaP import ManipPlanner, LMPlanner
from responsible_robot_bench.evaluator.config.rrb_manipulation_example import vlm_examples_baseline, llm_examples, vlm_examples_ablation
from responsible_robot_bench.main import logger

# Validate CLASS
class RRB_ManipulationEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None

    def load_demonstration(self):
        # Load case
        all_examples = {}
        # If only language PROMPT
        if self.config['language_only'] == 1:
            # visual icl baseline
            if self.config['visual_icl'] == 1:
                # Get VLM PROMPT case
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_ablation[variation.split('_')[0]]
            # language only
            else:
                # Get LLM PROMPT case
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = llm_examples[variation.split('_')[0]]
        # If using language + vision
        else:
            # main baseline
            if self.config['detection_box'] == 1 and self.config['multiview'] == 0 and self.config['visual_icl'] == 0 and self.config['multistep'] == 0:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_baseline[variation.split('_')[0]]
            # ablation study
            else:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_ablation[variation.split('_')[0]]

        return all_examples
    # Store Episode results
    def save_episode_metric(self, episode_info):
        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            # Store Episode results
            json.dump(episode_info, f, ensure_ascii=False)
    
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
    
    def print_task_eval_results(self, filename):
        folder_path = f"{self.log_path}/results"
        total_number_of_task = 0
        success_number_of_task = 0
        planner_steps = 0
        output_format_error = 0

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
                    if data["planner_output_error"] > 0:    
                        output_format_error += 1
                    # If the task succeeds
                    if task_success == 1:
                        success_number_of_task += 1
                    # Get Planner steps
                    planner_steps += data["planner_steps"]
                    total_number_of_task += 1

        # Store LOG: task count, success rate, etc.
        task_log = {}
        task_log['save_path'] = self.log_path
        task_log["total_num_tasks"] = total_number_of_task  # Total task count
        task_log["num_success"] = success_number_of_task    # Successful task count
        task_log["success_rate"] = success_number_of_task / total_number_of_task    # Success rate
        task_log["avg_planner_steps"] = planner_steps / total_number_of_task    # Average Planner steps
        task_log["output_format_error"] = output_format_error               # Number of output format errors

        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            # Store LOG
            json.dump(task_log, f, ensure_ascii=False)

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'action_success': []}
            image_history = []

            # Reset Virtual Environment: The environment OBS contains the coordinates of objects, as well as object properties, robot information, and camera information.
            _, obs, high_level_description_configs = self.env.reset()
            pass  # Removed an interactive debug checkpoint.
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
            avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(vars(copy.deepcopy(obs)), self.env.task_class, camera_views)
            
            # If not only using language, process images and add annotations.
            if not self.config['language_only']:
                for i, img_path in enumerate(img_path_list):
                    if 'front_rgb' in img_path:
                        img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
            # Draw BB bounding box
            if self.config['detection_box'] and not self.config['language_only']:
                img_path_list = draw_bounding_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list)
            # Multi-step image
            if self.config['multistep']:
                image_history.append(img_path_list[0])
            # Environment's this Episode's languagePrompt
            user_instruction = self.env.episode_language_instruction
            # Print this Episode's languagePrompt
            print(f"Instruction: {user_instruction}")
            # Reset Planner
            self.planner.reset()
            print("rrb_manipulation_evaluator.py : Finsihed resetting planner")
            done = False
            reasoning_list = [] # Store Planner's multi-round output

            # init_task using self.env.
            waypoints = self.env.task._task.get_waypoints()
            if len(waypoints) == 0:
                raise NoWaypointsError(
                    'No waypoints were found.', test_env.task)
            self.env.task._task.init_task()
            self.env.task._task.init_conditions()
            self.env.task._static_positions = True
            
            # If not finished
            while not done:
                # Call Planner Act to predict actions and Reasoning analysis
                print("rrb_manipulation_evaluator.py : Start: Calling planner.act")
                # Input the current average coordinates of the object
                pass  # Removed an interactive debug checkpoint.

                if self.config['traj_type'] == "pre-defined":
                    # load current environment status
                    object_list = high_level_description_configs.objects_list
                    available_skills = high_level_description_configs.high_level_descriptions
                # If not a pre-defined skill, object_list and available_skills are None.
                # If using vision, object information does not need to be included.
                # Need to input current scene information into the model if vision is not used.
                else:
                    object_list = None
                    available_skills = None
                # and user's instructions, get Planner's output
                if self.config['traj_type'] == "code":
                    # Predict code
                    if self.config['multistep']:
                        action, reasoning = self.planner.act(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                    else:
                        action, reasoning = self.planner.act(img_path_list, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                elif self.config['traj_type'] == "point":
                    # predict point, execute
                    if self.config['multistep']:
                        action, reasoning = self.planner.act(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                    else:
                        action, reasoning = self.planner.act(img_path_list, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                elif self.config['traj_type'] == "pre-defined":
                    assert object_list is not None, "object_list is None"
                    assert available_skills is not None, "available_skills is None"
                    # if using predefined skills, use Planner's predefined skill function
                    if self.config['multistep']:
                        action, reasoning = self.planner.act_from_pre_defined_skills(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation, object_list, available_skills)
                    else:
                        action, reasoning = self.planner.act_from_pre_defined_skills(img_path_list, user_instruction, str(avg_obj_coord), self.env.current_task_variation, object_list, available_skills)
                pass  # Removed an interactive debug checkpoint.
                # action result represents the midpoint in pixel space
                print(f"Planner Output Action: {action}")
                # store this round's Reasoning analysis
                reasoning_list.append(reasoning)
                # if action 0, stop directly
                if len(action) == 0:    # if no action, indicates parsing error, need to store reasoning information, check later
                    episode_info['reward'].append(0)
                    episode_info['action_success'].append(0)
                    info = {'task_success': 0, 'episode_elapsed_seconds': 0}
                    break
                # If there is an action, execute it one by one
                else:
                    for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:

                        # Input action, get environment feedback
                        # Action is a 7-dimensional vector, discrete action

                        # Use planner to define affordance map, other maps use default map


                        if self.config['traj_type'] == "pre-defined":
                            obs, reward, reward_safety, done, info = self.env.step_high_level_action(action_single)
                            # self.last_frame_obs, reward_act, reward_safety, terminate, info
                        else:
                            obs, reward, done, info = self.env.step(action_single)
                        print(f"Executed action: {action_single}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        # Update INFO to Planner
                        # Collect the actions executed in this round and the environment's feedback
                        self.planner.update_info(info)
                        
                        # Update the current image
                        img_path_list = self.env.save_image(camera_views)
                        for img_path in img_path_list:
                            if self.config['multistep']:
                                image_history.append(img_path)
                        
                        # Collect the reward for this round and whether the action was successful
                        episode_info['reward'].append(reward)
                        episode_info['action_success'].append(info['action_success'])
                        if done:
                            break
                

                avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(copy.deepcopy(obs), self.env.task_class, camera_views)
                if not done:    # Continue processing an active episode.
                    # Add coordinate markers to the updated image.
                    if not self.config['language_only']:
                        for i, img_path in enumerate(img_path_list):
                            if 'front_rgb' in img_path:
                                img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                    if self.config['detection_box'] and not self.config['language_only']:
                        img_path_list = draw_bounding_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list)
                        if self.config['multistep']:
                            if image_history[-1].split('.png')[0] in img_path_list[0]:
                                image_history.pop()
                                image_history.append(img_path_list[0])
            
            # evaluation metrics
            # Collect information about this Episode
            episode_info['instruction'] = user_instruction
            episode_info['avg_reward'] = np.mean(episode_info['reward'])    # Average reward
            episode_info.update(terminal_labels(self.env.task._task))             # Task success
            episode_info['num_steps'] = self.env._current_step              # Steps
            episode_info['planner_steps'] = self.planner.planner_steps          # Planner steps
            episode_info['planner_output_error'] = self.planner.output_json_error       # Planner output error
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]       # Episode time
            self.save_episode_metric(episode_info)  # Store Episode results
            # Store Planner output
            self.save_planner_outputs(reasoning_list)   # Store reasoning output
            progress_bar.update()
        self.print_task_eval_results(filename="summary.json")
        self.env.close()
    
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
            
            if 'exp_name' not in self.config or self.config['exp_name'] is None:
                # Define the log path
                self.log_path = 'running/rrb_manipulation/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}/{}'.format(
                                                                                                    real_model_name, 
                                                                                                    self.config['n_shots'], 
                                                                                                    self.config['resolution'], 
                                                                                                    self.config['detection_box'],
                                                                                                    self.config['multiview'],
                                                                                                    self.config['multistep'],
                                                                                                    self.config['visual_icl'],
                                                                                                    self.eval_set)
            else:
                # Define the log path
                self.log_path = 'running/rrb_manipulation/{}/{}/{}'.format(real_model_name, self.config["exp_name"], self.eval_set)
            # Create a virtual environment
            self.env = RRBManEnv(eval_set=self.eval_set, img_size=(self.config['resolution'], self.config['resolution']), down_sample_ratio=self.config["down_sample_ratio"], log_path=self.log_path, headless=self.config["headless"])
            # Get the case sample description inside the PROMPT and analyze the results
            if self.config['traj_type'] == "pre-defined" or self.config['traj_type'] == "code" or self.config['traj_type'] == "point":
                ic_examples = None # Test skipping this part
            else:
                ic_examples = self.load_demonstration()
            # loading correct system prompt
            if self.config['traj_type'] == "pre-defined":
                # use pre-define skill
                from responsible_robot_bench.evaluator.prompts_predefine_traj.rrb_manipulation_predefine_traj import PROMPT_API, PROMPT_EXAMPLES, PROMPT_LMP, PROMPT_LMP_WITH_COG, CURRENT_TASK_API
                # Compare whether to add Cognition
                system_prompt_input = PROMPT_LMP_WITH_COG
            elif self.config['traj_type'] == "code":
                from responsible_robot_bench.evaluator.prompts_sap.rrb_manipulation_sap_system_prompts import PROMPT_API, PROMPT_EXAMPLES, PROMPT_RLF, PROMPT_LMP, PROMPT_ISP, PROMPT_ISP_WITH_REF, PROMPT_LMP_WITH_COG 
                system_prompt_input = PROMPT_LMP_WITH_COG
            else:
                system_prompt_input = rrb_manipulation_system_prompt

            if self.config['traj_type'] == "pre-defined":
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
            elif self.config['traj_type'] == "point" or self.config['traj_type'] == "code":
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
                                            traj_type=self.config["traj_type"])              # Operation type

            # Start evaluation
            self.evaluate()
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
    parser.add_argument('--n_shots', type=int, default=10)
    parser.add_argument('--multiview', type=int, default=0)
    parser.add_argument('--detection_box', type=int, default=1)
    parser.add_argument('--multistep', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default="baseline")
    parser.add_argument('--visual_icl', type=int, default=0)
    parser.add_argument('--tp', type=int, default=1, help='number of tensor parallel splits of the model parameters')
    parser.add_argument('--headless', type=int, default=1, help='')
    parser.add_argument('--traj_type', type=str, default="pre-defined", help='code or pre-defined or point')
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
        'selected_indexes': [0], #[0, 12],
        'headless': args.headless,
        'traj_type': args.traj_type,
    }
    print("printing config ...")
    for config_key in config:
        print(f"{config_key}: {config[config_key]}")
    evaluator = RRB_ManipulationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()