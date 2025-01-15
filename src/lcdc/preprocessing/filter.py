from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..utils import datetime_to_sec
from ..vars import Variability, TableCols as TC
from .preprocessor import Preprocessor


class Filter(Preprocessor, ABC):

    @abstractmethod
    def condition(self, record: dict) -> bool:
        pass

    def __call__(self, record: dict) -> List[dict]:
        if self.condition(record):
            return [record]
        return []

class FilterFolded(Filter):

    def __init__(self, k=100, threshold=0.5):
        self.k = k
        self.threshold = threshold

    def condition(self, record: dict) -> bool:
        period = record[TC.PERIOD]

        folded = np.zeros(self.k)
        t = record[TC.DATA][:,0]
        phases = t / (period + 1e-10)
        indices = np.round((phases - np.floor(phases)) * self.k).astype(int)
        for idx in range(self.k):
            x, = np.where(indices == idx)
            if len(x) > 0:
                folded[idx] = record[TC.DATA][x,1].mean()

        return np.sum(folded != 0) / self.k >= self.threshold

class FilterMinLength(Filter):

    def __init__(self, length, step=None):
        self.length = length
        self.step = step

    def condition(self, record: dict) -> bool:
        if self.step is None:
            return len(record[TC.DATA]) >= self.length

        indices = set()
        for t in record[TC.DATA][:,0]:
            idx = np.round(t/self.step).astype(int)
            indices.add(idx)
            if len(indices) >= self.length:
                return True

        return False

class FilterByPeriodicity(Filter):

    def __init__(self, *types: Variability):
        self.filter_types = types

    def condition(self, record: dict) -> bool:
        if Variability.PERIODIC in self.filter_types and record[TC.PERIOD] != 0:
            return True
        return record[TC.VARIABILITY] in self.filter_types

class FilterByStartDate(Filter):

    def __init__(self, year, month, day, hour=0, minute=0, sec=0):
        date = f"{year}-{month}-{day}"
        time = f"{hour}:{minute}:{sec}"
        self.sec = datetime_to_sec(f'{date} {time}')

    def condition(self, record: dict) -> bool:
        return datetime_to_sec(record[TC.TIMESTAMP]) >= self.sec

class FilterByEndDate(FilterByStartDate):

    def condition(self, record: dict) -> bool:
        return datetime_to_sec(record[TC.TIMESTAMP]) <= self.sec

class FilterByNorad(Filter):

    def __init__(self, norad_list: List[int]):
        self.indices = set(norad_list)

    def condition(self, record: dict) -> bool:
        return int(record[TC.NORAD_ID]) in self.indices