from pathlib import Path
from responsible_robot_bench.evaluator.utils import read_txt
import os, re

prompt_base_path = Path(__file__).resolve().parent

PROMPT_get_affordance_map = read_txt(os.path.join(prompt_base_path, 'get_affordance_map_prompt.txt'))
PROMPT_get_avoidance_map = read_txt(os.path.join(prompt_base_path, 'get_avoidance_map_prompt.txt'))
PROMPT_get_gripper_map = read_txt(os.path.join(prompt_base_path, 'get_gripper_map_prompt.txt'))
PROMPT_get_rotation_map = read_txt(os.path.join(prompt_base_path, 'get_rotation_map_prompt.txt'))
PROMPT_get_velocity_map = read_txt(os.path.join(prompt_base_path, 'get_velocity_map_prompt.txt'))
PROMPT_parse_query_obj = read_txt(os.path.join(prompt_base_path, 'parse_query_obj_prompt.txt'))
PROMPT_planner = read_txt(os.path.join(prompt_base_path, 'planner_prompt.txt'))
PROMPT_composer = read_txt(os.path.join(prompt_base_path, 'composer_prompt.txt'))

PROMPT_sys_prompt = read_txt(os.path.join(prompt_base_path, 'general_system_prompt.txt'))
PROMPT_sys_prompt_wo_EXAMPLE = PROMPT_sys_prompt.replace("{PROMPT_EXAMPLES}", "")

def get_n_shot_prompt(prompt, agent_name, n_shots, output_type="string"):
    # The planner selects n_shots after this function returns the example list.
    available_agent_names = ["get_affordance_map", "get_avoidance_map",
                             "get_gripper_map", "get_rotation_map",
                             "get_velocity_map", "parse_query_obj",
                             "planner", "composer"]
    if agent_name not in available_agent_names:
        return prompt
    if output_type not in {"string", "list"}:
        raise ValueError("output_type must be either 'string' or 'list'")
    if output_type == "string":
        return prompt
    # Markers inside generated Python strings are not example boundaries.
    marker = "objects =" if agent_name in {"planner", "parse_query_obj"} else "# Query:"
    return [marker + item for item in re.split(r'(?m)^' + re.escape(marker), prompt)[1:]]
