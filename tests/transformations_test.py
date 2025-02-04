import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.preprocessing import Fold, ToGrid, DropColumns, ToApparentMagnitude
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

class TestToGrid(unittest.TestCase):
    def test_to_grid(self):
        record = {TC.TIME: np.array([0, 1, 2, 3, 4]), TC.MAG: np.array([1, 2, 3, 4, 5])}
        to_grid = ToGrid(sampling_frequency=1, size=5)
        gridded_record = to_grid.transform(record)
        self.assertEqual(len(gridded_record[TC.TIME]), 5)

class TestDropColumns(unittest.TestCase):
    def test_drop_columns(self):
        record = {TC.TIME: np.array([0, 1, 2, 3, 4]), 'extra_col': np.array([1, 2, 3, 4, 5])}
        drop_columns = DropColumns(columns=['extra_col'])
        transformed_record = drop_columns.transform(record)
        self.assertNotIn('extra_col', transformed_record)

class TestToApparentMagnitude(unittest.TestCase):
    def test_to_apparent_magnitude(self):
        record = {TC.MAG: np.array([1, 2, 3, 4, 5]), TC.PHASE: np.array([10, 20, 30, 40, 50]), TC.DISTANCE: np.array([1000, 1000, 1000, 1000, 1000])}
        to_apparent_magnitude = ToApparentMagnitude(inplace=False)
        transformed_record = to_apparent_magnitude.transform(record)
        self.assertIn('apparent_mag', transformed_record)

if __name__ == '__main__':
    unittest.main()


