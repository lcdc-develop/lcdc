from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..utils import datetime_to_sec
from ..vars import Variability, TableCols as TC
from .preprocessor import Preprocessor


class Filter(Preprocessor, ABC):
    """
    Filter class is an abstract class for filtering records based on some specified condition.
    """

    @abstractmethod
    def condition(self, record: dict) -> bool:
        """
        Abstract method to be implemented by subclasses to define the condition for filtering records.
        """
        pass

    def __call__(self, record: dict) -> List[dict]:
        """
        Calls the condition method to filter the record.

        Parameters:
        - record (dict): The record to be filtered.        

        Returns:
        - if condition(record) -> [record]
        - else -> []

        """
        if self.condition(record):
            return [record]
        return []

class FilterFolded(Filter):
    """
    Filters records if its folded light curve has a certain threshold of non-zero values.

    Args:
        k (int, optional): The size of folded light curve. Defaults to 100.
        threshold (float, optional): Minimal ratio of nonzero measurements in the folded light curve. Defaults to 0.5.
    """

    def __init__(self, k=100, threshold=0.5):
        self.k = k
        self.threshold = threshold

    def condition(self, record: dict) -> bool:
        period = record[TC.PERIOD]

        folded = np.zeros(self.k)
        t = record[TC.TIME]
        phases = t / (period + 1e-10)
        indices = np.round((phases - np.floor(phases)) * self.k).astype(int)
        for idx in range(self.k):
            x, = np.where(indices == idx)
            if len(x) > 0:
                folded[idx] = record[TC.MAG][x].mean()

        return np.sum(folded != 0) / self.k >= self.threshold

class FilterMinLength(Filter):
    """
    Filters records if the light curve length is at least `length`.

    Args:
        length (int): Minimal length of the light curve.
        step (float, optional): Time step used for length computation is `None` the number of measurements is used as the length . Defaults to None
    """


    def __init__(self, length, step=None):
        self.length = length
        self.step = step

    def condition(self, record: dict) -> bool:
        if self.step is None:
            return len(record[TC.TIME]) >= self.length

        indices = set()
        for t in record[TC.TIME]:
            idx = np.round(t/self.step).astype(int)
            indices.add(idx)
            if len(indices) >= self.length:
                return True

        return False

class FilterByPeriodicity(Filter):
    """
    Filters records based on the variability type.

    Args:
        *types (Variability): lcdc.vars.Variability types to filter.
    """

    def __init__(self, *types: Variability):
        self.filter_types = types

    def condition(self, record: dict) -> bool:
        if Variability.PERIODIC in self.filter_types and record[TC.PERIOD] != 0:
            return True
        return record[TC.VARIABILITY] in self.filter_types

class FilterByStartDate(Filter):
    """
    Filters out records created before the specified start date.

    Args:
        year (int): The year of the start date.
        month (int): The month of the start date.
        day (int): The day of the start date.
        hour (int, optional): The hour of the start date. Defaults to 0.
        minute (int, optional): The minute of the start date. Defaults to 0.
        sec (int|float, optional): The second of the start date. Defaults to 0.
    """

    def __init__(self, year, month, day, hour=0, minute=0, sec=0):
        date = f"{year}-{month}-{day}"
        time = f"{hour}:{minute}:{sec}"
        self.sec = datetime_to_sec(f'{date} {time}')

    def condition(self, record: dict) -> bool:
        return datetime_to_sec(record[TC.TIMESTAMP]) >= self.sec

class FilterByEndDate(FilterByStartDate):
    """
    Filters out records created after the specified start date.

    Args:
        year (int): The year of the start date.
        month (int): The month of the start date.
        day (int): The day of the start date.
        hour (int, optional): The hour of the start date. Defaults to 0.
        minute (int, optional): The minute of the start date. Defaults to 0.
        sec (int|float, optional): The second of the start date. Defaults to 0.
    """

    def condition(self, record: dict) -> bool:
        return datetime_to_sec(record[TC.TIMESTAMP]) <= self.sec

class FilterByNorad(Filter):
    """
    Filters records based on the NORAD ID.

    Args:
        norad_list (List[int]): List of allowed NORAD IDs.
    """

    def __init__(self, norad_list: List[int]):
        self.indices = set(norad_list)

    def condition(self, record: dict) -> bool:
        return int(record[TC.NORAD_ID]) in self.indices