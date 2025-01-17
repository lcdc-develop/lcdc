import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.preprocessing import Fold
from lcdc.vars import TableCols as TC

class FoldTest(unittest.TestCase):

    def test_fold(self):
        record = {
            TC.PERIOD: 3,
            TC.TIME: np.array([0,1,4,5,6]),
            TC.MAG: np.arange(5),
            TC.PHASE: np.ones(5),
            TC.DISTANCE: np.ones(5),
            TC.FILTER: np.ones(5)
        }

        r = Fold()(record)
        print(r[0][TC.TIME] == (np.array([0,0,1,1,2]))/3)

        self.assertAlmostEqual(np.abs(r[0][TC.TIME] - (np.array([0,0,1,1,2]))/3).sum(), 0)


