from functools import partial, reduce

from typing import List
from abc import ABC, abstractmethod

class Preprocessor(ABC):

    @abstractmethod
    def __call__():
        pass

class CustomProcessor(Preprocessor):
    """
    CustomProcessor class is a wrapper for custom functions that are not part of the predefined
    preprocessing steps. Ensure the correct output format of the function.

    Args:
        fun (function): The custom function to be applied to the record.
    """

    def __init__(self, fun) -> None:
        self.fun = fun

    def __call__(self, record: dict):
        res = self.fun(record)
        if isinstance(res, list):
            return res
        return [res]

class Compose(Preprocessor):

    def __init__(self, *funs: Preprocessor) -> None:
        self.funs = funs

    def __call__(self, record: dict):
        records = [record]
        for f in self.funs:
            if (records := reduce(list.__add__, map(f, records))) == []:
                break

        return records
