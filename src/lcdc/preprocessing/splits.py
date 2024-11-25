from abc import abstractmethod
import math
from typing import List

import numpy as np

from ..utils import RSO, Track
from .preprocessor import Preprocessor

class Split(Preprocessor):

    @abstractmethod
    def _find_split_indices(self, track: Track, object: RSO):
        pass

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        indices = self._find_split_indices(track, object)
        start = 0
        parts = []
        ends = indices + [len(track.data)]
        for i, arr in enumerate(np.split(track.data, indices)):
            t = Track(**track.__dict__)
            t.data = arr
            t.start_idx = start + track.start_idx
            t.end_idx = ends[i] + track.start_idx - 1
            start = t.end_idx + 1
            parts.append(t)
            
        return parts

class SplitByGaps(Split):

    def __init__(self, max_length=None):
        self.max_length = max_length
    
    def _find_split_indices(self, track: Track, object: RSO):
        data = track.data
        time_diff = data[1:,0] - data[:-1,0]
        split_indices, = np.where(time_diff > track.period)

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
    
    def _find_split_indices(self, track: Track, object: RSO):
        if track.period == 0:
            return [] 
        
        return SplitBySize(track.period * self.multiple)._find_split_indices(track, object)

class SplitBySize(Split):

    def __init__(self, max_length, uniform=False):
        self.max_length = max_length
        self.uniform = uniform

    def _find_split_indices(self, track: Track, object: RSO):

        split_indices = []
        length = track.data[-1,0] - track.data[0,0] + 1
        max_length = self.max_length
        if length > max_length:
            if self.uniform:
                max_length = (length / math.ceil(length / max_length))

            i = 0
            while i < len(track.data):
                start_idx = i
                t_start = track.data[start_idx,0]

                while i < len(track.data) and track.data[i,0] - t_start < max_length:
                    i += 1

                if i < len(track.data):
                    split_indices.append(i)

        return split_indices
