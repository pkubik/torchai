import random
import numpy as np
from collections import namedtuple, deque, Callable

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, record: Transition):
        self.buffer.append(record)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DecayingBinaryRandom:

    def __init__(self, eps_start=1.0, eps_end=0.01, eps_decay=1000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def eps_threshold(self, decay_steps: int) -> float:
        decay_factor = np.exp(-1. * decay_steps / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * decay_factor

    def sample(self, decay_steps: int) -> bool:
        return random.random() < self.eps_threshold(decay_steps)


def parse_log_record(record: str) -> dict:
    """
    >>> r = parse_log_record("Episode:  328 | Duration:    17 | Mean duration:  106.670 | Epsilon threshold: 0.01")
    >>> r == {"Episode": "328", "Duration": "17", "Mean duration": "106.670", "Epsilon threshold": "0.01"}
    True
    """
    fields = record.split('|')
    pairs = [field.split(':') for field in fields]
    return {key.strip(): value.strip() for key, value in pairs}


class SequenceCollector:
    def __init__(self, predicate: Callable):
        self.predicate = predicate
        self.sequences = []
        self._current_sequence = None

    def push(self, item):
        if self.predicate(item):
            if self._current_sequence is None:
                self._current_sequence = []
            self._current_sequence.append(item)
        else:
            if self._current_sequence is not None:
                self.sequences.append(self._current_sequence)
                self._current_sequence = None

    def close(self):
        if self._current_sequence is not None:
            self.sequences.append(self._current_sequence)
            self._current_sequence = None
