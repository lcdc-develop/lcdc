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

    def __call__(self, track: Track, object: RSO) -> List[Track]:
        tracks = [track]
        for f in self.funs:
            f = partial(f, object=object)
            if (tracks := reduce(list.__add__, map(f, tracks))) == []:
                break

        return tracks