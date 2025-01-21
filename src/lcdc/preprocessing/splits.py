from abc import abstractmethod
import math
from typing import List

import numpy as np

from ..vars import TableCols as TC, DATA_COLS
from ..utils import sec_to_datetime, datetime_to_sec 
from .preprocessor import Preprocessor

class Split(Preprocessor):
    """
    Split class is an abstract class for splitting records into parts based on some methodology.
    """

    @abstractmethod
    def _find_split_indices(self, record: dict):
        """
        Abstract method to be implemented by subclasses to define the indices to split the light curve at.
        """
        pass

    def __call__(self, record: dict):
        """
        Splits the record into parts based on the split indices.
        """
        indices = self._find_split_indices(record)
        parts = []
        start = 0
        for end in indices + [len(record[TC.TIME])]:
            r = record.copy()
            for c in filter(lambda x: x in r, DATA_COLS):
                r[c] = record[c][start:end].copy()
            r[TC.TIME] -= r[TC.TIME][0]
            r[TC.RANGE] = (start+record[TC.RANGE][0], end-1+record[TC.RANGE][0])
            parts.append(r)
            start = end
            
        return parts

class SplitByGaps(Split):
    """
    SplitByGaps splits the light curve into parts if the time 
    difference between two consecutive measurements is greater
    than the apparent rotational period.

    For non-periodic light curves, the split wont be performed.

    Args:
        max_length (int, optional): The maximum length of a part in seconds. Defaults to None.
                                    If its defined, multiple parts can be merged together 
                                    if their sum of lengths is less than max_length.
    """

    def __init__(self, max_length=None):
        self.max_length = max_length
    
    def _find_split_indices(self, record: dict):
        time = record[TC.TIME]
        time_diff = time[1:] - time[:-1]
        split_indices, = np.where(time_diff > record[TC.PERIOD])

        if self.max_length is not None:  # connect parts if sum of lengths is less than max_length
            beginnings = time[np.concatenate(([0], split_indices+1))]
            endings = time[np.concatenate((split_indices, [len(time)-1]))]
            part_dist = beginnings[1:] - endings[:-1] 
            part_len = endings - beginnings
            start, end = 0,0
            length = part_len[end]
            split = []

            while end < len(part_len):
                if start == end:
                    if length >= self.max_length:
                        split.append(end)
                        start += 1
                        length = part_len[start]
                    end += 1

                else:
                    length += part_dist[end-1] + part_len[end]

                    if length >= self.max_length:
                        split.append(end-1)
                        start = end
                        length = part_len[end]
                    else:
                        end += 1


            split_indices = split_indices[split]
        
        return list(split_indices + 1)

class SplitByRotationalPeriod(Split):
    """
    SplitByRotationalPeriod splits the light curve into individual rotational periods.
    For non-periodic light curves, the split wont be performed.

    Args:
        multiple (int, optional): The light curve is split by `multitiple` times 
                                    the apparent rotational period. Defaults to 1.
    """

    def __init__(self, multiple=1):
        self.multiple = multiple
    
    def _find_split_indices(self, record: dict):
        if record[TC.PERIOD] == 0:
            return [] 
        
        return SplitBySize(record[TC.PERIOD] * self.multiple)._find_split_indices(record)

class SplitBySize(Split):
    """
    SplitBySize splits the light curve into parts of a fixed size.

    Args: 
        max_length (int): The maximum length of a part in seconds.
        uniform (bool, optional): If True, the parts will have the same length <= max_length. Defaults to False.
    """

    def __init__(self, max_length, uniform=False):
        self.max_length = max_length
        self.uniform = uniform

    def _find_split_indices(self, record: dict):

        split_indices = []
        size = len(record[TC.TIME])
        length = record[TC.TIME][-1] - record[TC.TIME][0] + 1
        max_length = self.max_length
        if length > max_length:
            if self.uniform:
                max_length = (length / math.ceil(length / max_length))

            i = 0
            while i < len(record[TC.TIME]):
                start_idx = i
                t_start = record[TC.TIME][start_idx]

                while i < size and record[TC.TIME][i] - t_start < max_length:
                    i += 1

                if i < size:
                    split_indices.append(i)

        return split_indices
