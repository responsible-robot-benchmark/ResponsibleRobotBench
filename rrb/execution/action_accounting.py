"""Count ordinary actions when the dispatched native entry point is reached."""
from contextlib import contextmanager
from contextvars import ContextVar


_action = ContextVar('rrb_native_action_accounting', default=None)


@contextmanager
def account_native_action(environment, kind):
    """Associate one native call with its owning episode, without changing it."""
    token = _action.set({'environment': environment, 'kind': kind, 'entered': False})
    try:
        yield
    finally:
        _action.reset(token)


def record_native_action_entry(kind):
    """Charge once at the native execution boundary, including failed calls."""
    current = _action.get()
    if current is not None and current['kind'] == kind and not current['entered']:
        current['environment'].robot_actions_executed += 1
        current['entered'] = True
