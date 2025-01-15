from functools import partial, reduce

from typing import List
from abc import ABC, abstractmethod

from ..utils import Track, RSO

class Preprocessor(ABC):

    @abstractmethod
    def __call__() -> List[Track]:
        pass

class Compose(Preprocessor):

    def __init__(self, *funs: Preprocessor) -> None:
        self.funs = funs

    def __call__(self, record: dict) -> List[Track]:
        records = [record]
        for f in self.funs:
            if (records := reduce(list.__add__, map(f, records))) == []:
                break

        return records