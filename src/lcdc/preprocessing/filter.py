from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..utils import RSO, Track, datetime_to_sec
from ..vars import Variability
from .preprocessor import Preprocessor



class Filter(Preprocessor, ABC):

    @abstractmethod
    def condition(self, track: Track, object: RSO) -> bool:
        pass

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        if self.condition(track, object):
            return [track]
        return []

class FilterFolded(Filter):

    def __init__(self, k=100, threshold=0.5):
        self.k = k
        self.threshold = threshold

    def condition(self, track: Track, object: RSO) -> bool:
        period = track.period

        folded = np.zeros(self.k)
        t = track.data[:,0]
        phases = t / (period + 1e-10)
        indices = np.round((phases - np.floor(phases)) * self.k).astype(int)
        for idx in range(self.k):
            x, = np.where(indices == idx)
            if len(x) > 0:
                folded[idx] = track.data[x,1].mean()

        return np.sum(folded != 0) / self.k >= self.threshold

class FilterMinLength(Filter):

    def __init__(self, length, step=None):
        self.length = length
        self.step = step

    def condition(self, track: Track, object: RSO) -> bool:
        if self.step is None:
            return len(track.data) >= self.length

        indices = set()
        for t in track.data[:,0]:
            idx = np.round(t/self.step).astype(int)
            indices.add(idx)
            if len(indices) >= self.length:
                return True

        return False

class FilterByPeriodicity(Filter):

    def __init__(self, *types: Variability):
        self.filter_types = types

    def condition(self, track: Track, object: RSO) -> bool:
        if Variability.PERIODIC in self.filter_types and track.period != 0:
            return True
        return object.variability in self.filter_types

class FilterByStartDate(Filter):

    def __init__(self, year, month, day, hour=0, minute=0, sec=0):
        date = f"{year}-{month}-{day}"
        time = f"{hour}:{minute}:{sec}"
        self.sec = datetime_to_sec(f'{date} {time}')

    def condition(self, track: Track, object: RSO) -> bool:
        return datetime_to_sec(track.timestamp) >= self.sec

class FilterByEndDate(FilterByStartDate):

    def condition(self, track: Track, object: RSO) -> bool:
        return datetime_to_sec(track.timestamp) <= self.sec

class FilterByNorad(Filter):

    def __init__(self, norad_list: List[int]):
        self.indices = set(norad_list)

    def condition(self, track: Track, object: RSO) -> bool:
        return int(object.norad_id) in self.indices