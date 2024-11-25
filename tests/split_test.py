import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.utils import Track, RSO
from lcdc.preprocessing import (
    SplitByGaps, SplitBySize, SplitByRotationalPeriod
)


class TestSplitByGaps(unittest.TestCase):

    def test_call(self):
        track = Track(0,0,0,0,5,0,-1)
        t = np.array([0,1,2,3,6,7,8,9])
        track.data = np.zeros((len(t),5))
        track.data[:,0] = t
        track.period = 2
        split_by_gaps = SplitByGaps()

        result = split_by_gaps(track, MagicMock())
        assert len(result) == 2, f"Wrong number of splits: Expected 2, got {len(result)}"
        assert np.all(result[0].data[:,0] == t[:4]), f"Wrong split: Expected {t[:4]}, got {result[0].data[:,0]}"
        assert np.all(result[1].data[:,0] == t[4:]), f"Wrong split: Expected {t[:4]}, got {result[0].data[:,0]}"
        assert result[0].end_idx == 3, f"Wrong end_idx: Expected 3, got {result[0].end_idx}"
        assert result[1].start_idx == 4, f"Wrong start_idx: Expected 4, got {result[1].start_idx}"

class TestSplitBySize(unittest.TestCase):

    def _get_track(self):
        track = Track(0,0,0,0,5,0,-1)
        t = np.array([0,1,2,3,6,7,8,9])
        track.data = np.zeros((len(t),5))
        track.data[:,0] = t
        track.period = 2
        return track

    def test_basic(self):
        track = self._get_track()

        split_by_size = SplitBySize(max_length=4, uniform=False)
        result = split_by_size(track, MagicMock())
        assert len(result) == 2, f"Wrong number of splits: Expected 2, got {len(result)}"

        split_by_size = SplitBySize(max_length=2, uniform=False)
        result = split_by_size(track, MagicMock())
        assert len(result) == 4, f"Wrong number of splits: Expected 4, got {len(result)}"
        start = 0
        for r, s in zip(result, [2,2,2,2]):
            assert r.start_idx == start, f"Wrong start_idx: Expected {start}, got {r.start_idx}"
            assert len(r.data) == s, f"Wrong length: Expected {s}, got {len(r.data)}"
            start += s
    
    def test_uniform(self):
        track = self._get_track()

        split_by_size = SplitBySize(max_length=3, uniform=True)
        result = split_by_size(track, MagicMock())
        assert len(result) == 4, f"Wrong number of splits: Expected 4, got {len(result)}"

class TestSplitByRotationalPeriod(unittest.TestCase):
    
    def test_simple(self):
        track = Track(0,0,0,0,5,0,-1)
        t = np.array([0,1,2,3,6,7,8,9])
        track.data = np.zeros((len(t),5))
        track.data[:,0] = t
        track.period = 2
        split_by_rotational_period = SplitByRotationalPeriod(multiple=1)

        result = split_by_rotational_period(track, MagicMock())

        for r, s in zip(result, [2,2,2,2]):
            assert len(r.data) == s, f"Wrong length: Expected {s}, got {len(r.data)}"


        track.period = 3
        split_by_rotational_period = SplitByRotationalPeriod(multiple=1)

        result = split_by_rotational_period(track, MagicMock())

        for r, s in zip(result, [3,1,3,1]):
            assert len(r.data) == s, f"Wrong length: Expected {s}, got {len(r.data)}"
        
    def test_multiple(self):
        track = Track(0,0,0,0,5,0,-1)
        t = np.array([0,1,2,3,6,7,8,9])
        track.data = np.zeros((len(t),5))
        track.data[:,0] = t
        track.period = 1
        split_by_rotational_period = SplitByRotationalPeriod(multiple=2)

        result = split_by_rotational_period(track, MagicMock())

        for r, s in zip(result, [2,2,2,2]):
            assert len(r.data) == s, f"Wrong length: Expected {s}, got {len(r.data)}"


        track.period = 1
        split_by_rotational_period = SplitByRotationalPeriod(multiple=3)

        result = split_by_rotational_period(track, MagicMock())

        for r, s in zip(result, [3,1,3,1]):
            assert len(r.data) == s, f"Wrong length: Expected {s}, got {len(r.data)}"