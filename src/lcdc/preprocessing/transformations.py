from typing import List
from abc import abstractmethod

import numpy as np

from ..vars import DATA_COLS, TableCols as TC
from ..utils import fold, to_grid
from .preprocessor import Preprocessor

class Transformator(Preprocessor):

    @abstractmethod
    def transform(self, record: dict):
        pass

    def __call__(self, record: dict) -> List[dict]:
        return [self.transform(record)]
    

class Fold(Transformator):

    def transform(self, record: dict):
        record = fold(record, record[TC.PERIOD])
        return record
    
class ToGrid(Transformator):
    
    def __init__(self, sampling_frequency: float, size: int):
        self.frequency = sampling_frequency
        self.size = size
    
    def transform(self, record: dict):

        record = to_grid(record, self.frequency)
        #
        some = list(filter(lambda x: x in record, DATA_COLS))[0]
        if len(record[some]) < self.size:
            for c in filter(lambda x: x in record, DATA_COLS):
                record[c] = np.concatenate([record[c], np.zeros(self.size - len(record[c]))])

        if len(record[some]) > self.size:
            for c in filter(lambda x: x in record, DATA_COLS):
                record[c] = record[c][:self.size]

        return record

class DropColumns(Transformator):
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def transform(self, record: dict):
        for c in self.columns:
            del record[c]
        return record
    