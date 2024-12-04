from typing import List

import numpy as np

from ..utils import RSO, Track, fold_track
from .preprocessor import Preprocessor

class Fold(Preprocessor):

    def __call__(self, t: Track, object: RSO) -> List[Track]:

        data = fold_track(t.data, t.period)

        new_track = Track(**t.__dict__)
        new_track.data = data

        return [new_track]
    
#TODO: TO GRID