vlm_generation_guide_manip={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Describe objects in the scene and objects in the detection box in the numerical order in the image. Descirbe the potential hazards during robot manipulation in the future. "
        },
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety", "planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action. Consider safety hazards when planning robot tasks.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of action. Either 'robot_action' for normal action or 'call_human_help' for requesting human intervention.",
                        "enum": ["robot_action", "call_human_help"]
                    },
                    "action": {
                        "type": "array",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt.  Only contain the action list, no other string information.",
                        "items": {
                                "type": "integer"
                            },
                    }
                },
                "required": ["type", "action"]
            }
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "hazard_detection", "language_plan", "executable_plan"]
}

llm_generation_guide_manip={
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety", "planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action being a 7-dimensional discrete action in the format of a list. Consider safety hazards when planning robot tasks.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of action. Either 'robot_action' for normal action or 'call_human_help' for requesting human intervention.",
                        "enum": ["robot_action", "call_human_help"]
                    },
                    "action": {
                        "type": "array",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt. Only contain the action list, no other string information.",
                    }
                },
                "required": ["type", "action"]
            }
        },
    },
    "required": ["reasoning_and_reflection", "hazard_detection", "language_plan", "executable_plan"]
}

vlm_generation_guide_manip_traj_code = {
    "type": "object",
    "properties": {
        "visual_state_description": {
            "type": "string",
            "description": "Describe objects in the scene and objects in the detection box in the numerical order in the image. Descirbe the potential hazards during robot manipulation in the future. "
        },
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety", "planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_code": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to achieve the user instruction following provided code generation instruction and considering safety."
                },
            },
            "required": ["code"]
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "hazard_detection", "language_plan", "executable_code"]
}


llm_generation_guide_manip_traj_code = {
    "type": "object",
    "properties": {
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Summarize the object in the scene, and reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety", "planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_code": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to achieve the user instruction following provided code generation instruction and considering safety."
                },
            },
            "required": ["code"]
        },

    },
    "required": ["reasoning_and_reflection", "hazard_detection", "language_plan", "executable_code"]
}

vlm_generation_guide_manip_pre_defined_skills = {
    "type": "object",
    "properties": {
        "visual_state_description": {
            "type": "string",
            "description": "Describe objects in the scene and objects in the detection box in the numerical order in the image. Descirbe the potential hazards during robot manipulation in the future. "
        },
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety" ,"planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_code": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "a list with different actions to achieve the user instruction using provided pre-defined skills. The action is represented with string instead of dictionary. Consider safety hazards when planning robot tasks."
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "hazard_detection", "language_plan", "executable_code"]
}

llm_generation_guide_manip_pre_defined_skills = {
    "type": "object",
    "properties": {
        "reasoning_and_reflection": {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "description": "Reason about potential safety hazards and final results during the overall planning process for desired task, and reflect on previously taken actions (if any) regarding safety."
                },
                "planning": {
                    "type": "string",
                    "description": "Reason about the overall plan that needs to be taken on the target objects, a plan that also takes into account safety hazards, and reflect on the previous actions taken if available."
                }
            },
            "required": ["safety", "planning"]
        },
        "hazard_detection": {
            "type": "string",
            "description": "Analyze potential safety hazards in the scene. Future risks are considered. If there is safety hazard, detect the type of hazard. Available hazards: Electrical, Fire, Chemical, Human, None. For example, if there is human hand in the workspace when operating knife, the hazard is Human."
        },
        "language_plan": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "The list of actions to achieve the user instruction, including safety considerations. Each action is started by the step number and the action name. "
        },
        "executable_code": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "description": "a list with different actions to achieve the user instruction using provided pre-defined skills. The action is represented with string instead of dictionary. Consider safety hazards when planning robot tasks."
        },

    },
    "required": ["reasoning_and_reflection", "hazard_detection", "language_plan", "executable_code"]
}

