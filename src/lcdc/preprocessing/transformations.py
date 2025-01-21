from typing import List
from abc import abstractmethod

import numpy as np

from ..vars import DATA_COLS, TableCols as TC
from ..utils import fold, to_grid
from .preprocessor import Preprocessor

class Transformator(Preprocessor):
    """
    Abstract class for transformations. Transformation modifies the original light curve within the record.
    """

    @abstractmethod
    def transform(self, record: dict):
        """
        Abstract method to be implemented by subclasses to define the transformation of the record.
        """
        pass

    def __call__(self, record: dict) -> List[dict]:
        """Call method for the transformation.

        Args:
            record (dict): record containing the light curve.

        Returns:
            [record]: List with single element, the transformed record. 
        """
        return [self.transform(record)]
    

class Fold(Transformator):
    """
    Fold class is a transformation folds the light curve by its
    apparent rotational period. Does not apply for non-variable 
    or aperiodic light curves.

    Influenced fields: `time`, `mag`, `phase`, `distance`, `filter`
    """

    def transform(self, record: dict):
        record = fold(record, record[TC.PERIOD])
        return record
    
class ToGrid(Transformator):
    """
    ToGrid class is a transformation that resamples the light curve 
    by `sampling_frequency`. The result is padded / truncated  to 
    a fixed size.

    Influenced fields: `time`, `mag`, `phase`, `distance`, `filter`

    Args:
        sampling_frequency (float): The resampling frequency [Hz].
        size (int): The fixed size of the resampled light curve.
    """
    
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
    """
    DropColumns removes field specified in the `columns` parameter.

    Args:
        columns (List[str]): List of fields to be removed
    """
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def transform(self, record: dict):
        for c in self.columns:
            del record[c]
        return record
    