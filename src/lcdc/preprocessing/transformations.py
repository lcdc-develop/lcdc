from typing import List

import numpy as np

from ..vars import TableCols as TC
from ..utils import fold_track, track_to_grid
from .preprocessor import Preprocessor

class Fold(Preprocessor):

    def __call__(self, record: dict):

        data = fold_track(record[TC.DATA], record[TC.PERIOD])

        record[TC.DATA] = data

        return [record]
    
class ToGrid(Preprocessor):
    
    def __init__(self, sampling_frequency: float, size: int):
        self.frequency = sampling_frequency
        self.size = size
    
    def __call__(self, record: dict):
        data = record[TC.DATA]

        record[TC.DATA] = track_to_grid(data, self.frequency)
        #
        if record[TC.DATA].shape[0]< self.size:
            new_data = np.zeros((self.size, record[TC.DATA].shape[1]))
            new_data[:record[TC.DATA].shape[0]] = record[TC.DATA]
            record[TC.DATA] = new_data
        if record[TC.DATA].shape[0] > self.size:
            record[TC.DATA] = record[TC.record[TC.DATA]][:self.size]

        return [record]
    