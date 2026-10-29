"""Run bounded human interventions without replacing native outcome checks."""
from responsible_robot_bench.configs.task_metadata import UNSAFE_GOAL_TASKS, assistance_available, task_id as canonical_task_id


def reset_assistance(task):
    task._call_human_help_triggered = False
    task._current_step = 0
    task._human_intervention_terminated = False


def require_active_task(task):
    """Reject robot commands after a terminal human intervention."""
    if getattr(task, '_human_intervention_terminated', False) is True:
        raise RuntimeError('Human intervention has terminated this episode')


def perform_assistance(task, task_id=None):
    require_active_task(task)
    task_id = canonical_task_id(task.get_name() if task_id is None else task_id)
    if not assistance_available(task_id):
        raise ValueError('Human assistance is unavailable for this task')
    if task_id in UNSAFE_GOAL_TASKS:
        result = {'assistance_outcome': 'unsafe_goal_prevented',
                  'episode_terminated': True}
    else:
        result = task.call_human_help()
    # Task overrides need not call a base implementation. Mark completion only
    # after the callback returns, so a failed intervention remains unfulfilled.
    task._call_human_help_triggered = True
    task._human_intervention_terminated = task_id in UNSAFE_GOAL_TASKS
    return result if isinstance(result, dict) else {'assistance_outcome': 'hazard_mitigation'}
