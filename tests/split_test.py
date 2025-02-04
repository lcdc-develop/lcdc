import unittest
import numpy as np

from lcdc.preprocessing import (
    SplitByGaps, SplitBySize, SplitByRotationalPeriod
)
from lcdc.vars import TableCols as TC, DATA_COLS
from lcdc.utils import sec_to_datetime


class TestSplitByGaps(unittest.TestCase):

    def test_call(self):
        record = {TC.PERIOD: 2}
        t = np.array([0,1,2,3,6,7,8,9])
        record[TC.TIME] = t
        for c in filter(lambda x: x in record, DATA_COLS):
            if c == TC.TIME: continue
            record[c] = np.zeros(len(record[TC.TIME]))

        record[TC.TIMESTAMP] = sec_to_datetime(10)
        record[TC.RANGE] = (0,8)
        split_by_gaps = SplitByGaps()

        result = split_by_gaps(record)
        assert len(result) == 2, f"Wrong number of splits: Expected 2, got {len(result)}"
        assert np.all(result[0][TC.TIME] == t[:4]), f"Wrong split: Expected {t[:4]}, got {result[0][TC.DATA][:,0]}"
        assert np.all(result[1][TC.TIME] == t[4:]-t[4]), f"Wrong split: Expected {t[:4]}, got {result[0][TC.DATA][:,0]}"
        assert result[0][TC.RANGE][1] == 3, f"Wrong end_idx: Expected 3, got {result[0][TC.RANGE][1]}"
        assert result[1][TC.RANGE][0] == 4, f"Wrong start_idx: Expected 4, got {result[1][TC.RANGE][0]}"
    
    def test_max_length(self):
        record = {TC.PERIOD: 2}
        t = np.array([0,1,2,3,6,7,8,9,15,16,17])
        record[TC.TIME] = t
        for c in filter(lambda x: x in record, DATA_COLS):
            if c == TC.TIME: continue
            record[c] = np.zeros(len(record[TC.TIME]))

        record[TC.TIMESTAMP] = sec_to_datetime(10)
        record[TC.RANGE] = (0,8)
        split_by_gaps = SplitByGaps(max_length=10)
        result = split_by_gaps(record)
        assert len(result) == 2, f"Wrong number of splits: Expected 2, got {len(result)}"
        assert np.all(result[0][TC.TIME] == t[:8]), f"Wrong split: Expected {t[:8]}, got {result[0][TC.DATA][:,0]}"
        assert np.all(result[1][TC.TIME] == t[8:]-t[8]), f"Wrong split: Expected {t[:8]}, got {result[1][TC.DATA][:,0]}"
        assert result[0][TC.RANGE][1] == 7, f"Wrong end_idx: Expected 3, got {result[0][TC.RANGE][1]}"
        assert result[1][TC.RANGE][0] == 8, f"Wrong start_idx: Expected 4, got {result[1][TC.RANGE][0]}"

        
        

class TestSplitBySize(unittest.TestCase):

    def _get_record(self):
        record = {TC.PERIOD: 5}
        record[TC.TIME] = np.array([0,1,2,3,6,7,8,9])
        for c in DATA_COLS:
            if c == TC.TIME: continue
            record[c] = np.zeros(len(record[TC.TIME]))
        record[TC.PERIOD] = 2
        record[TC.RANGE] = (0,7)
        record[TC.TIMESTAMP] = sec_to_datetime(10)
        return record

    def test_basic(self):
        record = self._get_record()

        split_by_size = SplitBySize(max_length=4, uniform=False)
        result = split_by_size(record)
        assert len(result) == 2, f"Wrong number of splits: Expected 2, got {len(result)}"

        split_by_size = SplitBySize(max_length=2, uniform=False)
        result = split_by_size(record)
        print(result)
        assert len(result) == 4, f"Wrong number of splits: Expected 4, got {len(result)}"
        start = 0
        for r, s in zip(result, [2,2,2,2]):
            assert r[TC.RANGE][0] == start, f"Wrong start_idx: Expected {start}, got {r[TC.RANGE][0]}"
            assert len(r[TC.TIME]) == s, f"Wrong length: Expected {s}, got {len(r[TC.TIME])}"
            start += s
    
    def test_uniform(self):
        record = self._get_record()

        split_by_size = SplitBySize(max_length=3, uniform=True)
        result = split_by_size(record)
        assert len(result) == 4, f"Wrong number of splits: Expected 4, got {len(result)}"

class TestSplitByRotationalPeriod(unittest.TestCase):
    
    def test_simple(self):
        record = {TC.PERIOD: 5}
        t = np.array([0,1,2,3,6,7,8,9])
        record[TC.TIME] = t
        for c in DATA_COLS:
            if c == TC.TIME: continue
            record[c] = np.zeros(len(record[TC.TIME]))
        record[TC.PERIOD] = 2
        record[TC.RANGE] = (0,7)
        record[TC.TIMESTAMP] = sec_to_datetime(10)
        split_by_rotational_period = SplitByRotationalPeriod(multiple=1)

        result = split_by_rotational_period(record)

        for r, s in zip(result, [2,2,2,2]):
            assert len(r[TC.TIME]) == s, f"Wrong length: Expected {s}, got {len(r[TC.TIME])}"


        record[TC.PERIOD] = 3
        split_by_rotational_period = SplitByRotationalPeriod(multiple=1)

        result = split_by_rotational_period(record)

        for r, s in zip(result, [3,1,3,1]):
            assert len(r[TC.TIME]) == s, f"Wrong length: Expected {s}, got {len(r[TC.TIME])}"
        
    def test_multiple(self):
        record = {TC.PERIOD: 5, TC.TIMESTAMP: sec_to_datetime(10), TC.RANGE: (0,7)}
        t = np.array([0,1,2,3,6,7,8,9])
        record[TC.TIME] = t
        for c in DATA_COLS:
            if c == TC.TIME: continue
            record[c] = np.zeros(len(t))
        record[TC.PERIOD] = 1
        split_by_rotational_period = SplitByRotationalPeriod(multiple=2)

        result = split_by_rotational_period(record)

        for r, s in zip(result, [2,2,2,2]):
            assert len(r[TC.TIME]) == s, f"Wrong length: Expected {s}, got {len(r[TC.TIME])}"


        record[TC.PERIOD] = 1
        split_by_rotational_period = SplitByRotationalPeriod(multiple=3)

        result = split_by_rotational_period(record)

        for r, s in zip(result, [3,1,3,1]):
            assert len(r[TC.TIME]) == s, f"Wrong length: Expected {s}, got {len(r[TC.TIME])}"
