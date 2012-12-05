import json
import numpy as np
from collections import Counter
import random
import itertools

def convert_timestamps(fh): 

    times = json.load(fh)
    times_counter = Counter(times)
    max_count = max(times_counter.values())
    delta = 1.0/(max_count)

    def breakup_bin(bin_val, bin_count):
        return [
            bin_val+(i*delta) 
            for i in random.sample(range(max_count), bin_count)
        ]

    click_times = list(
        itertools.chain(
            *itertools.starmap(breakup_bin, times_counter.iteritems())
        )
    )

    click_times.sort()
    assert len(click_times) == len(set(click_times))

    time_range = np.arange(min(click_times), max(click_times)+delta, delta)

    Y = np.zeros(len(time_range))

    Y[time_range.searchsorted(click_times)] = 1

    return time_range, Y, delta
