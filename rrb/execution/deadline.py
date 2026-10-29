"""Interrupt a finite episode while leaving time to record simulator labels."""
import math
import signal
import threading
from contextlib import contextmanager


class EpisodeDeadlineExceeded(TimeoutError):
    """The configured episode wall-time budget was exhausted."""


class _TransportDeadlineInterrupt(BaseException):
    """Cross a transport that catches and retries ordinary exceptions."""


def is_model_request_timeout(error):
    """Identify the provider's request timeout without conflating episode limits."""
    try:
        from openai import APITimeoutError
    except ImportError:
        return False
    return isinstance(error, APITimeoutError)


class EpisodeDeadline:
    def __init__(self, seconds=None):
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('Episode timeout must be finite and positive')
        self.seconds = seconds
        self.previous = None
        self.expired = False

    def start(self):
        if self.seconds is None:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Episode deadlines require a dedicated main-thread process')
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError('Another real-time timer is already active')
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def _expired(self, signum, frame):
        self.expired = True
        raise EpisodeDeadlineExceeded('Episode wall-time limit reached')

    def check(self):
        # Some retained simulator callbacks catch all exceptions. An expired
        # deadline must still terminate the episode when control returns.
        if self.expired:
            raise EpisodeDeadlineExceeded('Episode wall-time limit reached')

    @contextmanager
    def protect_transport(self):
        """Preserve the episode alarm across SDK exception and retry handlers."""
        if self.previous is None:
            self.check()
            yield
            return
        self.check()
        previous_handler = signal.getsignal(signal.SIGALRM)

        def interrupt(signum, frame):
            self.expired = True
            raise _TransportDeadlineInterrupt()

        signal.signal(signal.SIGALRM, interrupt)
        try:
            try:
                yield
            except _TransportDeadlineInterrupt as error:
                raise EpisodeDeadlineExceeded('Episode wall-time limit reached') from error
            self.check()
        finally:
            signal.signal(signal.SIGALRM, previous_handler)

    def cancel(self):
        if self.previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
            self.previous = None
