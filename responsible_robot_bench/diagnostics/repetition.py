"""Static repetition diagnostics for original planner responses.

The historical filename and ``check_repetition_errors`` API are retained.
Repeated text/actions are observations, not proof of hallucination, invalid API
use, task failure, or an unnecessary retry. Execution feedback is needed to
interpret them. This module never executes generated code or normalizes case.
"""
import ast
from collections import Counter
import json
import math
import re


def _region_end(text, start):
    """Skip a malformed JSON object without treating its children as replies."""
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def _responses(content):
    """Read complete JSON replies, including fenced/concatenated log entries."""
    if isinstance(content, dict):
        return [(content, None)]
    if isinstance(content, (list, tuple)):
        result = []
        for item in content:
            result.extend(_responses(item))
        return result
    if not isinstance(content, str):
        return []
    decoder = json.JSONDecoder()
    result, position = [], 0
    while position < len(content):
        start = content.find('{', position)
        if start < 0:
            break
        try:
            response, end = decoder.raw_decode(content, start)
        except (ValueError, RecursionError):
            # Preserve the attempt's position but do not infer actions from a
            # broken response, quoted reasoning, or its partial child objects.
            end = _region_end(content, start)
            result.append((None, content[start:end]))
            position = end
        else:
            result.append((response, None))
            position = end
    return result


def _literal(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, RecursionError):
            pass
    return value


def _freeze(value):
    """Hash exact parsed values without conflating bool/int or changing case."""
    if isinstance(value, dict):
        return ('dict', tuple(sorted((str(key), _freeze(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ('sequence', tuple(_freeze(item) for item in value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite action value')
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    raise ValueError('Unsupported action value')


def _actions(plan, key):
    """Extract list entries or top-level Python statements without evaluation."""
    plan = _literal(plan)
    if isinstance(plan, dict) and 'code' in plan:
        plan = _literal(plan['code'])
    if isinstance(plan, (list, tuple)) and len(plan) == 1 and isinstance(plan[0], str):
        nested = _literal(plan[0])
        if isinstance(nested, (list, tuple)):
            plan = nested
    if isinstance(plan, (list, tuple)):
        result = []
        for action in plan:
            if isinstance(action, (list, tuple)) and len(action) == 1 and isinstance(action[0], dict):
                action = action[0]
            if isinstance(action, dict):
                if action.get('type') == 'call_human_help':
                    action = 'call_human_help'
                elif 'action' in action:
                    action = action['action']
                else:
                    raise ValueError('Unrecognized action wrapper')
            if key == 'executable_plan':
                action = _literal(action)
                if action != 'call_human_help' and not (
                    isinstance(action, (list, tuple)) and len(action) == 7 and
                    all(type(value) in (int, float) and math.isfinite(value) for value in action)):
                    raise ValueError('Expected a seven-value pose')
            elif not isinstance(action, str):
                raise ValueError('Expected a skill string')
            result.append((_freeze(action), repr(action)))
        return result
    if isinstance(plan, str) and key == 'executable_code':
        tree = ast.parse(plan)
        # Do not unroll loops or flatten branches: textual occurrences are not
        # an execution trace, and mutually exclusive calls may never repeat.
        return [(('statement', ast.dump(node, include_attributes=False)), ast.unparse(node))
                for node in tree.body]
    raise ValueError('Unsupported executable plan')


def _malformed_text_patterns(content):
    """Inspect repeated prose in JSON strings without recovering a partial plan."""
    patterns = []
    for text in re.findall(r'"((?:\\.|[^"\\])*)(?:"|$)', content, re.DOTALL):
        sentences = [sentence.strip() for sentence in re.findall(r'[^.!?]+[.!?]', text)]
        for sentence, count in Counter(sentences).items():
            if count > 3 and len(sentence) >= 40 and len(re.findall(r'\b[A-Za-z]+\b', sentence)) >= 8:
                patterns.append(dict(kind='repeated_text', text=sentence, count=count))
    return patterns


def repetition_diagnostics(content):
    """Return one explicit diagnostic per complete/invalid response region.

    A candidate is an exact action appearing more than three times, or an
    entire action sequence consisting of two or more copies of one block.
    These inherited thresholds are heuristic. Even a detected candidate has
    ``proves_hallucination=False``; no action, API, object, or hazard validity
    can be inferred from it. Invalid plans are never recovered or action-scored.
    Malformed JSON can separately retain exact repeated prose as text evidence.
    """
    diagnostics = []
    for attempt, (response, raw) in enumerate(_responses(content), 1):
        result = dict(attempt=attempt, status='unscored', patterns=[], repeated=False,
                      proves_hallucination=False, semantics='static_repetition_candidate')
        diagnostics.append(result)
        if response is None:
            result['status'] = 'invalid_response'
            result['patterns'] = _malformed_text_patterns(raw)
            result['repeated'] = bool(result['patterns'])
            if result['repeated']:
                result['semantics'] = 'malformed_response_text_repetition_candidate'
            continue
        body = response.get('properties', response)
        if not isinstance(body, dict):
            result['status'] = 'invalid_response'
            continue
        key = next((key for key in ('executable_code', 'executable_plan') if key in body), None)
        if key is None:
            result['status'] = 'missing_plan'
            continue
        result['plan_field'] = key
        try:
            actions = _actions(body[key], key)
        except (ValueError, TypeError, SyntaxError, RecursionError, OverflowError):
            result['status'] = 'invalid_plan'
            continue
        result.update(status='scored', action_count=len(actions))
        tokens = [token for token, _ in actions]
        counts = Counter(tokens)
        for token, count in counts.items():
            if count > 3:
                positions = [index for index, value in enumerate(tokens) if value == token]
                result['patterns'].append(dict(kind='frequent_action', count=count,
                                               positions=positions, action=actions[positions[0]][1]))
        length = len(tokens)
        for size in range(1, length // 2 + 1):
            if length % size == 0 and all(tokens[index] == tokens[index % size] for index in range(length)):
                result['patterns'].append(dict(kind='complete_sequence', block_length=size,
                                               repetitions=length // size))
                break
        result['repeated'] = bool(result['patterns'])
    return diagnostics


def partial_code_repetition_diagnostics(code):
    """Detect repeated complete top-level statements in invalid Code output.

    This is a text diagnostic for an incomplete response, never a replacement
    for parsing or executing the program. Indented blocks, comments, fences,
    and incomplete lines are not treated as standalone executed actions.
    """
    if not isinstance(code, str):
        return []
    lines = []
    for line in code.splitlines():
        if not line or line != line.lstrip() or line.startswith(('#', '```')):
            continue
        try:
            tree = ast.parse(line)
        except (SyntaxError, ValueError, RecursionError):
            continue
        if len(tree.body) != 1:
            continue
        node = tree.body[0]
        if isinstance(node, ast.Assign) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            lines.append(line)
    if not lines:
        return []
    result = repetition_diagnostics({'executable_code': '\n'.join(lines)})
    for item in result:
        item['status'] = 'scored_partial_code'
        item['semantics'] = 'partial_code_repetition_candidate'
    return [item for item in result if item['repeated']]


def check_repetition_errors(content):
    """Compatibility result: candidate messages by attempt and affected count.

    The historical word 'errors' is an API name, not a claim of hallucination.
    Consumers must keep this diagnostic separate from verified failures.
    """
    attempts = {}
    for diagnostic in repetition_diagnostics(content):
        if not diagnostic['repeated']:
            continue
        messages = []
        for pattern in diagnostic['patterns']:
            if pattern['kind'] == 'frequent_action':
                message = f"Action appears {pattern['count']} times: {pattern['action']}"
            elif pattern['kind'] == 'repeated_text':
                message = f"Text appears {pattern['count']} times: {pattern['text']}"
            else:
                message = (f"Complete block of {pattern['block_length']} actions appears "
                           f"{pattern['repetitions']} times")
            messages.append(message + '; repetition candidate, hallucination not established')
        attempts[diagnostic['attempt']] = messages
    return attempts, len(attempts)
