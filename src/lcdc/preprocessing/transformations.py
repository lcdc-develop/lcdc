from abc import abstractmethod
from typing import List

import numpy as np
from ..utils import fold, to_grid, correct_standard_magnitude
from ..vars import DATA_COLS, TableCols as TC
from .preprocessor import Preprocessor


class Transformator(Preprocessor):
    """
    Abstract class for transformations. Transformation modifies the original light curve within the record.
    """

    @staticmethod
    @abstractmethod
    def transform(record: dict):
        """
        Abstract method to be implemented by subclasses to define the transformation of the record.
        """

    @classmethod
    def __call__(cls, record: dict) -> List[dict]:
        """Call method for the transformation.

        Args:
            record (dict): record containing the light curve.

        Returns:
            [record]: List with single element, the transformed record.
        """
        return [cls.transform(record)]


class Fold(Transformator):
    """
    Fold class is a transformation folds the light curve by its
    apparent rotational period. Does not apply for non-variable
    or aperiodic light curves.

    Influenced fields: `time`, `mag`, `phase`, `distance`, `filter`
    """

    @staticmethod
    def transform(record: dict):
        return fold(record, record[TC.PERIOD])

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
        some = [x for x in DATA_COLS if x in record][0]
        if len(record[some]) < self.size:
            for c in filter(lambda x: x in record, DATA_COLS):
                record[c] = np.concatenate([record[c], np.zeros(self.size - len(record[c]))])

        if len(record[some]) > self.size:
            for c in filter(lambda x: x in record, DATA_COLS):
                record[c] = record[c][: self.size]

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

class ToApparentMagnitude(Transformator):
    """
    ToApparentMagnitude class converts standardized magnitude (to distance 1000km and 90 deg phase angle)
    into apparent magnitude.

    Args:
        inplace (bool): If True, the original magnitude field is replaced with the corrected one.
                        Otherwise, the corrected magnitude is stored in a new field 'apparent_mag'.

    Source: McCue GA, Williams JG, Morford JM. Optical characteristics of artificial satellites.
        Planetary and Space Science. 1971;19(8):851-68.
        Available from: https://www.sciencedirect.com/science/article/pii/0032063371901371.
    """

    def __init__(self, inplace=True):
        self.inplace = inplace

    def transform(self, record):
        corrected_mag = correct_standard_magnitude(record[TC.MAG], record[TC.PHASE], record[TC.DISTANCE], beta=0.5)

        col = TC.MAG if self.inplace else "apparent_mag"
        record[col] = corrected_mag

        return record