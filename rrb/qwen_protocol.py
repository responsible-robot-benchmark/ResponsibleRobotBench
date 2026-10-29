"""Pure request and completion rules for the local Qwen chat endpoint."""
import math
import re


def stop_strings(value):
    if value is None:
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ValueError('stop must be a string or a list of 1 to 4 strings')
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError('stop strings must be nonempty strings')
    return values


def only_keys(value, allowed, location):
    if not isinstance(value, dict):
        raise ValueError(f'{location} must be an object')
    extra = value.keys() - allowed
    if extra:
        raise ValueError(f'Unsupported {location} fields: {", ".join(sorted(extra))}')


def validate_schema(schema, location='schema'):
    """Reject constraints that LMFE 0.11.3 would ignore or cannot enforce.

    This deliberately advertises a subset, including all original RRB schemas.
    Numeric bounds, combinators, references and regexes require another backend.
    """
    only_keys(schema, {'type', 'properties', 'required', 'additionalProperties',
                      'items', 'minItems', 'maxItems', 'minLength', 'maxLength',
                      'enum', 'description', 'title'}, location)
    kind = schema.get('type')
    if not isinstance(kind, str) or kind not in {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}:
        raise ValueError(f'{location}.type must name one supported JSON type')
    applicable = {'properties': 'object', 'required': 'object',
                  'additionalProperties': 'object', 'items': 'array',
                  'minItems': 'array', 'maxItems': 'array',
                  'minLength': 'string', 'maxLength': 'string'}
    for key, expected in applicable.items():
        if key in schema and kind != expected:
            raise ValueError(f'{location}.{key} requires type {expected}')
    for key in ('title', 'description'):
        if key in schema and not isinstance(schema[key], str):
            raise ValueError(f'{location}.{key} must be a string')
    for lower, upper in (('minItems', 'maxItems'), ('minLength', 'maxLength')):
        for key in (lower, upper):
            if key in schema and (type(schema[key]) is not int or schema[key] < 0):
                raise ValueError(f'{location}.{key} must be a nonnegative integer')
        if lower in schema and upper in schema and schema[lower] > schema[upper]:
            raise ValueError(f'{location}.{lower} exceeds {upper}')
    # LMFE 0.11.3 admits the first item of maxItems=0 arrays, and its string
    # token-cache shortcut treats maxLength=0 as unbounded. Fail before decoding.
    if schema.get('maxItems') == 0 or schema.get('maxLength') == 0:
        raise ValueError(f'{location}: zero maximum length is unsupported by this backend')
    if 'enum' in schema:
        values = schema['enum']
        valid = {'string': lambda x: isinstance(x, str),
                 'integer': lambda x: type(x) is int,
                 'number': lambda x: type(x) in (int, float) and math.isfinite(x)}
        if (kind not in valid or not isinstance(values, list) or not values
                or not all(valid[kind](item) for item in values)):
            raise ValueError(f'{location}.enum requires strings or numbers matching type')
        if 'minLength' in schema or 'maxLength' in schema:
            raise ValueError(f'{location}: enum with length constraints is unsupported')
    if kind == 'object':
        properties = schema.get('properties', {})
        if not isinstance(properties, dict):
            raise ValueError(f'{location}.properties must be an object')
        for key, child in properties.items():
            validate_schema(child, f'{location}.properties.{key}')
        required = schema.get('required', [])
        if (not isinstance(required, list) or any(not isinstance(key, str) for key in required)
                or len(set(required)) != len(required) or set(required) - properties.keys()):
            raise ValueError(f'{location}.required must list unique declared properties')
        if 'additionalProperties' in schema:
            # LMFE's dictionary and declared-property paths behave differently.
            extra = schema['additionalProperties']
            if extra is False and 'properties' in schema:
                pass
            elif isinstance(extra, dict) and 'properties' not in schema:
                validate_schema(extra, f'{location}.additionalProperties')
            else:
                raise ValueError(f'{location}.additionalProperties is unsupported in this form')
    if kind == 'array' and 'items' in schema:
        validate_schema(schema['items'], f'{location}.items')


def response_schema(value):
    """Return a decoding schema; None means unconstrained text."""
    if value is None:
        return None
    only_keys(value, {'type', 'json_schema'}, 'response_format')
    kind = value.get('type')
    if kind in ('text', 'json_object'):
        if 'json_schema' in value:
            raise ValueError('json_schema requires response_format.type=json_schema')
        return {'type': 'object'} if kind == 'json_object' else None
    if kind != 'json_schema':
        raise ValueError('response_format.type must be text, json_object or json_schema')
    spec = value.get('json_schema')
    only_keys(spec, {'name', 'description', 'strict', 'schema'}, 'json_schema')
    if not isinstance(spec.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', spec['name']):
        raise ValueError('json_schema.name must contain 1 to 64 letters, digits, underscores or hyphens')
    if 'description' in spec and not isinstance(spec['description'], str):
        raise ValueError('json_schema.description must be a string')
    if spec.get('strict') is not None and type(spec['strict']) is not bool:
        raise ValueError('json_schema.strict must be a boolean or null')
    schema = spec.get('schema')
    validate_schema(schema)
    return schema


def trim_stop(text, stops):
    positions = [position for stop in stops if (position := text.find(stop)) >= 0]
    return (text[:min(positions)], True) if positions else (text, False)


class CompletionStop:
    """Transformers stopping callable for one completion, excluding its prompt.

    Decode the generated prefix to handle stops split across tokenizer boundaries
    and stops embedded inside a token. The endpoint serves one sequence at a time.
    """
    def __init__(self, tokenizer, prompt_tokens, stops):
        self.tokenizer, self.prompt_tokens, self.stops = tokenizer, prompt_tokens, stops

    def __call__(self, input_ids, scores=None, **kwargs):
        text = self.tokenizer.decode(input_ids[0][self.prompt_tokens:],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        return trim_stop(text, self.stops)[1]


def completion_result(text, stops, token_ids, max_tokens, eos_token_ids):
    answer, stopped = trim_stop(text, stops)
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos = bool(token_ids) and token_ids[-1] in (eos_token_ids or [])
    reason = 'stop' if stopped or eos or len(token_ids) < max_tokens else 'length'
    return answer, reason
