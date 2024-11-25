import unittest
import random
from unittest.mock import MagicMock
import numpy as np

from lcdc.utils import Track, RSO
from lcdc.vars import Variability
from lcdc.preprocessing import (
    FilterFolded, FilterMinLength, FilterByEndDate, FilterByStartDate, FilterByPeriodicity,
    FilterByNorad
)

class TestFilterFolded(unittest.TestCase):

    def test_condition_true(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((5,5))
        track.data[:,0] = np.arange(5)
        track.data[:,1] = np.ones(5)
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertTrue(filter_folded.condition(track, MagicMock()))

    def test_condition_false(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((5,5))
        track.data[:,0] = np.arange(5)
        track.data[:,1] = np.zeros(5)
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertFalse(filter_folded.condition(track, MagicMock()))

class TestFilterMinLength(unittest.TestCase):

    def test_condition_true(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((5,5))
        track.data[:,0] = np.arange(5)
        filter_min_length = FilterMinLength(length=5)
        self.assertTrue(filter_min_length.condition(track, MagicMock()))

    def test_condition_false(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((2,5))
        track.data[:,0] = np.arange(2)
        filter_min_length = FilterMinLength(length=5)
        self.assertFalse(filter_min_length.condition(track, MagicMock()))
    
    def test_condition_true_step(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((5,5))
        track.data[:,0] = np.arange(5) * 2
        track.data[:,1] = np.ones(5)
        filter_min_length = FilterMinLength(length=5, step=2)
        self.assertTrue(filter_min_length.condition(track, MagicMock()))

        filter_min_length = FilterMinLength(length=2, step=5)
        self.assertTrue(filter_min_length.condition(track, MagicMock()))

    def test_condition_false_step(self):
        track = Track(0,0,0,0,5,0,-1)
        track.data = np.zeros((5,5))
        track.data[:,0] = np.arange(5) 
        track.data[:,1] = np.ones(5)
        filter_min_length = FilterMinLength(length=5, step=2)
        self.assertFalse(filter_min_length.condition(track, MagicMock()))

class TestFilterByEndDate(unittest.TestCase):
    
    def test_condition_true(self):
        track = Track(0,0,0,0,5,0,-1)
        track.timestamp = "2021-01-01 00:00:00"
        filter_end_date = FilterByEndDate(year=2021, month=2, day=1, hour=0, minute=0, sec=0)
        self.assertTrue(filter_end_date.condition(track, MagicMock()))
        filter_end_date = FilterByEndDate(year=2021, month=1, day=1, hour=0, minute=0, sec=1)
        self.assertTrue(filter_end_date.condition(track, MagicMock()))
    
    def test_condition_false(self):
        track = Track(0,0,0,0,5,0,-1)
        track.timestamp = "2021-02-01 00:00:00"
        filter_end_date = FilterByEndDate(year=2021, month=1, day=1, hour=0, minute=0, sec=0)
        self.assertFalse(filter_end_date.condition(track, MagicMock()))
        filter_end_date = FilterByEndDate(year=2021, month=1, day=31, hour=23, minute=59, sec=59)
        self.assertFalse(filter_end_date.condition(track, MagicMock()))

        
class TestFilterByStartDate(unittest.TestCase):
    
    def test_condition_false(self):
        track = Track(0,0,0,0,5,0,-1)
        track.timestamp = "2021-01-01 00:00:00"
        filter_end_date = FilterByStartDate(year=2021, month=2, day=1, hour=0, minute=0, sec=0)
        self.assertFalse(filter_end_date.condition(track, MagicMock()))
        filter_end_date = FilterByStartDate(year=2021, month=1, day=1, hour=0, minute=0, sec=1)
        self.assertFalse(filter_end_date.condition(track, MagicMock()))
    
    def test_condition_true(self):
        track = Track(0,0,0,0,5,0,-1)
        track.timestamp = "2021-02-01 00:00:00"
        filter_end_date = FilterByStartDate(year=2021, month=1, day=1, hour=0, minute=0, sec=0)
        self.assertTrue(filter_end_date.condition(track, MagicMock()))
        filter_end_date = FilterByStartDate(year=2021, month=1, day=31, hour=23, minute=59, sec=59)
        self.assertTrue(filter_end_date.condition(track, MagicMock()))

class TestFilterByPeriodicity(unittest.TestCase):
    
    def test_condition_true(self):
        rso = RSO(0,0,"", "", Variability.PERIODIC)

        for v in Variability:
            rso.variability = v
            filter_periodicity = FilterByPeriodicity(v)
            self.assertTrue(filter_periodicity.condition(MagicMock(), rso))

class TestFilterByNorad(unittest.TestCase):
    
    def test_condition_true(self):
        rso = RSO(0,1,"", "", Variability.PERIODIC)

        indices = [random.randint(0,100) for i in range(10)]
        indices.append(1)
        self.assertTrue(FilterByNorad(indices).condition(MagicMock(), rso))

    def test_condition_false(self):
        rso = RSO(0,1,"", "", Variability.PERIODIC)

        indices = [random.randint(2,100) for i in range(10)]
        self.assertFalse(FilterByNorad(indices).condition(MagicMock(), rso))