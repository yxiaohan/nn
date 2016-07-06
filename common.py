import numpy as np
from datetime import datetime


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def print_func_name(func):
    print(func.__name__)


class SpeedTest(object):
    def __init__(self):
        self._start_time = None

    def start(self):
        self._start_time = datetime.now()

    def stop(self):
        if self._start_time is None:
            raise ValueError('not started yet!')
        time_used = datetime.now() - self._start_time
        self._start_time = None
        print('time used:%s' % time_used)
        return time_used
