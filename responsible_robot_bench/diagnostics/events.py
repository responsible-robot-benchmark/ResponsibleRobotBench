"""Preserve observed diagnostic evidence and translate it to failure events."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import signal
import threading


_observations = ContextVar('rrb_action_diagnostic_observations', default=None)


@contextmanager
def commit_diagnostic_records():
    """Defer an episode alarm while committing a short in-memory journal update.

    Python dispatches signal handlers on the main thread even when another
    thread received the signal. Restore and invoke the previous handler after
    the commit. Simulator and planner calls must never be made inside it.
    """
    if not hasattr(signal, 'SIGALRM') or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    if not callable(previous):
        yield
        return
    pending = []
    def defer(signum, frame):
        pending.append((signum, frame))
    signal.signal(signal.SIGALRM, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous)
        if pending:
            previous(*pending[0])


@contextmanager
def observe_action_diagnostics():
    """Capture observations made during one action, including on interruption."""
    observations = []
    token = _observations.set(observations)
    try:
        yield observations
    finally:
        _observations.reset(token)


def record_diagnostic_observation(kind, evidence):
    """Record an observation batch; members of a batch have no known order."""
    observations = _observations.get()
    if observations is not None:
        observations.append(dict(kind=kind, observation_index=len(observations),
                                 evidence=deepcopy(evidence)))


def diagnostic_failure_events(info):
    """Return all additional codes, distinguishing observed and unknown order.

    The index describes when evidence was observed, not the exact physical
    onset. Contacts sampled in the same physics step share an index and are
    unordered. Legacy info without observation records has no cross-kind order.
    """
    events, seen_contacts = [], []
    seen_kinds = set()

    def add(kind, evidence, index=None):
        if kind == 'arm_contact_onset':
            code = 7
        elif kind in ('wrong_selected_target', 'wrong_grasped_object'):
            code = 3
        else:
            return
        event = dict(failure_code=code, kind=kind, evidence=evidence,
                     order_basis='observation' if index is not None else 'unknown')
        if index is not None:
            event['observation_index'] = index
            if kind == 'arm_contact_onset':
                event['within_observation_order'] = 'unordered'
        events.append(event)

    for observation in info.get('diagnostic_observations', []):
        kind, evidence = observation.get('kind'), observation.get('evidence')
        index = observation.get('observation_index')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            index = None
        if kind == 'arm_contact_onsets' and isinstance(evidence, list):
            for contact in evidence:
                if isinstance(contact, dict) and contact.get('event') == 'arm_contact_onset':
                    add('arm_contact_onset', contact, index)
                    seen_contacts.append(contact)
        elif kind == 'target_selection_diagnostic' and isinstance(evidence, dict):
            seen_kinds.add(kind)
            if evidence.get('status') == 'wrong_selected_target':
                add('wrong_selected_target', evidence, index)
        elif kind == 'grasp_target_diagnostic' and isinstance(evidence, dict):
            seen_kinds.add(kind)
            if evidence.get('status') == 'grasp_target_mismatch':
                add('wrong_grasped_object', evidence, index)

    # Older, mocked, or separately produced diagnostics may lack observation
    # indices. Preserve them without claiming they occurred in this list order.
    collision = info.get('collision_diagnostic')
    if isinstance(collision, dict):
        remaining = list(seen_contacts)
        for contact in collision.get('contact_events', []):
            if isinstance(contact, dict) and contact.get('event') == 'arm_contact_onset':
                if contact in remaining:
                    remaining.remove(contact)
                else:
                    add('arm_contact_onset', contact)
    selection = info.get('target_selection_diagnostic')
    if ('target_selection_diagnostic' not in seen_kinds and isinstance(selection, dict)
            and selection.get('status') == 'wrong_selected_target'):
        add('wrong_selected_target', selection)
    grasp = info.get('grasp_target_diagnostic')
    if ('grasp_target_diagnostic' not in seen_kinds and isinstance(grasp, dict)
            and grasp.get('status') == 'grasp_target_mismatch'):
        add('wrong_grasped_object', grasp)
    return events
