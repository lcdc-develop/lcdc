from typing import List

import numpy as np

from ..utils import RSO, Track, fold_track, track_to_grid
from .preprocessor import Preprocessor

class Fold(Preprocessor):

    def __call__(self, t: Track, object: RSO) -> List[Track]:

        data = fold_track(t.data, t.period)

        new_track = Track(**t.__dict__)
        new_track.data = data

        return [new_track]
    
class ToGrid(Preprocessor):
    
    def __init__(self, sampling_frequency: float, size: int):
        self.frequency = sampling_frequency
        self.size = size
    
    def __call__(self, t: Track, object: RSO) -> List[Track]:
        data = t.data

        t.data = track_to_grid(data, self.frequency)
        #
        if t.data.shape[0]< self.size:
            new_data = np.zeros((self.size, t.data.shape[1]))
            new_data[:t.data.shape[0]] = t.data
            t.data = new_data
        if t.data.shape[0] > self.size:
            t.data = t.data[:self.size]

        new_track = Track(**t.__dict__)
        return [new_track]
    