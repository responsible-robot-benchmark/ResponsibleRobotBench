import json
import os
import glob

def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

def load_saved_data(path):
    dirs = os.listdir(path)
    json_files = glob.glob(os.path.join(path, '*.json')) 
    instructions = []
    for json_file in json_files:
        with open(json_file, 'r+') as f:
            for line in f:  
                data = json.loads(line)
                instructions.append(data['instruction'])
                break
    return instructions


from enum import Enum
# Stable legacy IDs. Ordered episode lists may contain multiple occurrences of
# the same failure, successful action markers, and diagnostic-only code 8.
# Keep termination separate; never deduplicate events or overwrite earlier ones.
class FailureType(Enum):
    DEBUG = -1
    NO_ERROR = 0
    UNVALID_SKILL_OR_FUNCTION = 1  # Unknown/unavailable API; API names remain case-sensitive.
    UNVALID_OBJECT = 2              # The returned object is not in the environment
    MANIPULATION_FALSE_POSITIVE = 3  # Evidence-backed selected-target or grasp-target mismatch.
    
    FORMAT_ERROR = 4  # Format error, cannot parse
    TRAJ_PLANNING_ERROR = 5  # Trajectory planning error
    TRAJ_PLANNING_NOT_ARRIVED = 6  # Trajectory planning error
    COLLISION_ERROR = 7  # New non-gripper robot-arm/environment contact.

    REPEAT_CONTENT = 8  # Static repetition candidate, NOT proof of hallucination or task failure.
    HAZARD_DETECTION_FAILED_ERROR = 9  # Known reference hazard omitted/miscategorized; labels ignore case.
    HAZARD_DETECTION_FALSE_POSITIVE = 10  # Known predicted category differs from the reference (including None).
    # Wrong known categories can produce both 9 and 10. Missing/unknown labels
    # require an explicit unscored diagnostic, not an invented hazard category.
    TIMEOUT = 11  # Episode deadline after any preceding planning/execution failures.

    UNVALID_EXP = 99 # Invalid Experiment

class FailureException(Exception):
    def __init__(self, failure_type: FailureType, message: str = ""):
        self.failure_type = failure_type
        self.message = message or failure_type.name
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.failure_type.name}] {self.message}"
    

