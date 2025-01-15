from abc import abstractmethod
import math
from typing import List

import numpy as np

from ..vars import TableCols as TC
from ..utils import sec_to_datetime, datetime_to_sec 
from .preprocessor import Preprocessor

class Split(Preprocessor):

    @abstractmethod
    def _find_split_indices(self, record: dict):
        pass

    def __call__(self, record: dict):
        indices = self._find_split_indices(record)
        start = 0
        parts = []
        ends = indices + [len(record)]
        for i, arr in enumerate(np.split(record[TC.DATA], indices)):
            r = record.copy()
            r[TC.TIMESTAMP] = sec_to_datetime(datetime_to_sec(record[TC.TIMESTAMP]) + arr[0,0])
            r[TC.DATA] = arr
            r[TC.DATA][:,0] -= r[TC.DATA][0,0]
            r_start = start + record[TC.START_IDX]
            r_end =  ends[i] + record[TC.START_IDX] - 1
            r[TC.RANGE] = (r_start, r_end)
            start = record[TC.END_IDX] + 1
            parts.append(record)
            
        return parts

class SplitByGaps(Split):

    def __init__(self, max_length=None):
        self.max_length = max_length
    
    def _find_split_indices(self, record: dict):
        data = record[TC.DATA]
        time_diff = data[1:,0] - data[:-1,0]
        split_indices, = np.where(time_diff > record[TC.PERIOD])

        if self.max_length is not None:  # connect parts if sum of lengths is less than max_length
            beginnings = data[np.concatenate(([0], split_indices+1)),0]
            endings = data[np.concatenate((split_indices, [len(data)-1])),0]
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

    def __init__(self, multiple=1):
        self.multiple = multiple
    
    def _find_split_indices(self, record: dict):
        if record[TC.PERIOD] == 0:
            return [] 
        
        return SplitBySize(record[TC.PERIOD] * self.multiple)._find_split_indices(record)

class SplitBySize(Split):

    def __init__(self, max_length, uniform=False):
        self.max_length = max_length
        self.uniform = uniform

    def _find_split_indices(self, record: dict):

        split_indices = []
        length = record[TC.DATA][-1,0] - record[TC.DATA][0,0] + 1
        max_length = self.max_length
        if length > max_length:
            if self.uniform:
                max_length = (length / math.ceil(length / max_length))

            i = 0
            while i < len(record[TC.DATA]):
                start_idx = i
                t_start = record[TC.DATA][start_idx,0]

                while i < len(record[TC.DATA]) and record[TC.DATA][i,0] - t_start < max_length:
                    i += 1

                if i < len(record[TC.DATA]):
                    split_indices.append(i)

        return split_indices
