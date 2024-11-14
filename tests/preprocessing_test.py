import unittest
from unittest.mock import MagicMock
from utils import Track, RSO
import numpy as np

from dataset_builder.preprocessing import (
    FilterFolded, FilterMinLength, SplitByGaps, SplitBySize, ComputeAmplitude, Compose
)

class TestFilterFolded(unittest.TestCase):

    def test_condition_true(self):
        track = MagicMock()
        track.fold.return_value = np.array([1, 1, 1, 1, 1])
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertTrue(filter_folded.condition(track, MagicMock()))

    def test_condition_false(self):
        track = MagicMock()
        track.fold.return_value = np.array([0, 0, 0, 0, 0])
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertFalse(filter_folded.condition(track, MagicMock()))

class TestFilterMinLength(unittest.TestCase):

    def test_condition_true(self):
        track = MagicMock()
        track.length.return_value = 10
        filter_min_length = FilterMinLength(length=5)
        self.assertTrue(filter_min_length.condition(track, MagicMock()))

    def test_condition_false(self):
        track = MagicMock()
        track.length.return_value = 3
        filter_min_length = FilterMinLength(length=5)
        self.assertFalse(filter_min_length.condition(track, MagicMock()))

class TestSplitByGaps(unittest.TestCase):

    def test_call(self):
        track = MagicMock()
        track.split_by_gaps.return_value = [track, track]
        split_by_gaps = SplitByGaps(max_length=10)
        result = split_by_gaps(track, MagicMock())
        self.assertEqual(result, [track, track])

class TestSplitBySize(unittest.TestCase):

    def test_call(self):
        track = MagicMock()
        track.split_by_size.return_value = [track, track]
        split_by_size = SplitBySize(max_length=10, uniform=True)
        result = split_by_size(track, MagicMock())
        self.assertEqual(result, [track, track])

class TestComputeAmplitude(unittest.TestCase):

    def test_call(self):
        track = MagicMock()
        compute_amplitude = ComputeAmplitude()
        result = compute_amplitude(track, MagicMock())
        track.compute_amplitude.assert_called_once()
        self.assertEqual(result, [track])

class TestCompose(unittest.TestCase):

    def test_call(self):
        track = MagicMock()
        preprocessor1 = MagicMock()
        preprocessor2 = MagicMock()
        preprocessor1.return_value = [track]
        preprocessor2.return_value = [track]
        compose = Compose(preprocessor1, preprocessor2)
        result = compose(track, MagicMock())
        self.assertEqual(result, [track])

if __name__ == '__main__':
    unittest.main()