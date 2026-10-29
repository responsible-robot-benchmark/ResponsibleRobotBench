"""Parse and validate model responses before action dispatch."""
import ast
import json
from responsible_robot_bench.execution.poses import discrete_pose


class InvalidPlan(ValueError):
    pass


def parse_pose_response(text):
    """Validate discrete plans before any robot action is dispatched."""
    try:
        response = json.loads(text)
        entries = response.get('properties', response)['executable_plan']
        if isinstance(entries, str):
            entries = ast.literal_eval(entries)
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError('executable_plan must contain at least one action')
        actions = []
        for entry in entries:
            if isinstance(entry, dict) and entry.get('type') == 'call_human_help':
                actions.append('call_human_help')
                continue
            if isinstance(entry, (list, tuple)) and len(entry) == 1 and isinstance(entry[0], dict):
                entry = entry[0]
            action = entry['action'] if isinstance(entry, dict) else entry
            if isinstance(action, str):
                action = ast.literal_eval(action)
            if isinstance(action, tuple):
                action = list(action)
            discrete_pose(action)
            actions.append(action)
    except (ValueError, KeyError, AttributeError, TypeError, SyntaxError) as error:
        raise InvalidPlan(f'Invalid pose response: {error}') from error
    return actions, response


def parse_code_response(text):
    """Extract syntactically valid code; never replace malformed output with motion."""
    try:
        response=json.loads(text)
        properties=response.get('properties',response)
        code=properties['executable_code']
        if not isinstance(code,str) or not code.strip():
            raise ValueError('executable_code must be a nonempty string')
        ast.parse(code)
    except (ValueError,KeyError,AttributeError,TypeError,SyntaxError) as error:
        raise InvalidPlan(f'Invalid code response: {error}') from error
    return [code],response
