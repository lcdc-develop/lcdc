from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import List

import numpy as np
from utils import RSO, Track, Variability, datetime_to_sec, StrVars



class Preprocessor(ABC):

    @abstractmethod
    def __call__() -> List[Track]:
        pass

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
        return np.sum(track.fold(self.k) != 0) / self.k >= self.threshold

class FilterMinLength(Filter):

    def __init__(self, length, step=None):
        self.length = length
        self.step = step

    def condition(self, track: Track, object: RSO) -> bool:
        return track.length(self.step) >= self.length

class FilterByPeriodicity(Filter):

    def __init__(self, *types: Variability):
        self.filter_types = types

    def condition(self, track: Track, object: RSO) -> bool:
        return object.variability in self.filter_types

class FilterByStartDate(Filter):

    def __init__(self, year, month, day, hour=0, minute=0, sec=0):
        date = f"{year}-{month}-{day}"
        time = f"{hour}:{minute}:{sec}"
        self.sec = datetime_to_sec(date, time)

    def condition(self, track: Track, object: RSO) -> bool:
        return track.timestamp >= self.sec

class FilterByEndDate(FilterByStartDate):

    def condition(self, track: Track, object: RSO) -> bool:
        return track.timestamp <= self.sec

class FilterByNorad(FilterByStartDate):

    def __init__(self, norad_list: List[int]):
        self.indices = set(norad_list)

    def condition(self, track: Track, object: RSO) -> bool:
        return track.norad_id in self.indices

class SplitByGaps(Preprocessor):

    def __init__(self, max_length=None):
        self.max_length = max_length

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        return track.split_by_gaps(self.max_length)

class SplitBySize(Preprocessor):

    def __init__(self, max_length, uniform=False):
        self.max_length = max_length
        self.uniform = uniform

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        return track.split_by_size(self.max_length, self.uniform)

class ComputeAmplitude(Preprocessor):

    def __call__(self, track: Track, object: RSO) -> Track:
        track.compute_amplitude()
        return [track]

class ComputeFourierSeries(Preprocessor):

    def __init__(self, order):
        self.order = order
    
    def __call__(self, track: Track, object: RSO) -> Track:
        coefs, _ = track.get_fourier_coeficients(self.order)
        track.stats[StrVars.FOURIER_COEFS] = list(coefs)
        return [track]


class Compose(Preprocessor):

    def __init__(self, *funs: Preprocessor) -> None:
        self.funs = funs

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        tracks = [track]
        for f in self.funs:
            f = partial(f, object=object)
            if (tracks := reduce(list.__add__, map(f, tracks))) == []:
                break

        return tracks