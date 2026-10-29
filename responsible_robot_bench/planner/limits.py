"""Validate service context limits against the loaded model configuration."""


def context_limit(requested, native_limit):
    if type(native_limit) is not int or native_limit < 1024:
        raise ValueError('Model context limit must be a positive integer of at least 1024')
    if requested is None:
        return native_limit
    if isinstance(requested, str) and requested.isdecimal():
        requested = int(requested)
    if type(requested) is not int or not 1024 <= requested <= native_limit:
        raise ValueError('Service context limit must be between 1024 and the native model limit')
    return requested


def fits_context(prompt_tokens, output_tokens, limit):
    if any(type(value) is not int or value < 1 for value in [prompt_tokens, output_tokens, limit]):
        raise ValueError('Token counts and context limit must be positive integers')
    return prompt_tokens + output_tokens <= limit
