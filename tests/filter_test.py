import unittest
import random
import numpy as np

from lcdc.vars import Variability, TableCols as TC
from lcdc.preprocessing import (
    FilterFolded, FilterMinLength, FilterByEndDate, FilterByStartDate, FilterByPeriodicity,
    FilterByNorad
)

class TestFilterFolded(unittest.TestCase):

    def test_condition_true(self):
        record = {TC.PERIOD: 5}
        record[TC.DATA] = np.zeros((5,5))
        record[TC.DATA][:,0] = np.arange(5)
        record[TC.DATA][:,1] = np.ones(5)
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertTrue(filter_folded.condition(record))

    def test_condition_false(self):
        record = {TC.PERIOD: 10}
        record[TC.DATA] = np.zeros((5,5))
        record[TC.DATA][:,0] = np.arange(5)
        record[TC.DATA][:,1] = np.zeros(5)
        filter_folded = FilterFolded(k=5, threshold=0.5)
        self.assertFalse(filter_folded.condition(record))

class TestFilterMinLength(unittest.TestCase):

    def test_condition_true(self):
        record = {}
        record[TC.DATA] = np.zeros((5,5))
        record[TC.DATA][:,0] = np.arange(5)
        filter_min_length = FilterMinLength(length=5)
        self.assertTrue(filter_min_length.condition(record))

    def test_condition_false(self):
        record = {}
        record[TC.DATA] = np.zeros((2,5))
        record[TC.DATA][:,0] = np.arange(2)
        filter_min_length = FilterMinLength(length=5)
        self.assertFalse(filter_min_length.condition(record))
    
    def test_condition_true_step(self):
        record = {}
        record[TC.DATA] = np.zeros((5,5))
        record[TC.DATA][:,0] = np.arange(5) * 2
        record[TC.DATA][:,1] = np.ones(5)
        filter_min_length = FilterMinLength(length=5, step=2)
        self.assertTrue(filter_min_length.condition(record))

        filter_min_length = FilterMinLength(length=2, step=5)
        self.assertTrue(filter_min_length.condition(record))

    def test_condition_false_step(self):
        record = {}
        record[TC.DATA] = np.zeros((5,5))
        record[TC.DATA][:,0] = np.arange(5) 
        record[TC.DATA][:,1] = np.ones(5)
        filter_min_length = FilterMinLength(length=5, step=2)
        self.assertFalse(filter_min_length.condition(record))

class TestFilterByEndDate(unittest.TestCase):
    
    def test_condition_true(self):
        record = {}
        record[TC.TIMESTAMP] = "2021-01-01 00:00:00"
        filter_end_date = FilterByEndDate(year=2021, month=2, day=1, hour=0, minute=0, sec=0)
        self.assertTrue(filter_end_date.condition(record))
        filter_end_date = FilterByEndDate(year=2021, month=1, day=1, hour=0, minute=0, sec=1)
        self.assertTrue(filter_end_date.condition(record))
    
    def test_condition_false(self):
        record = {}
        record[TC.TIMESTAMP] = "2021-02-01 00:00:00"
        filter_end_date = FilterByEndDate(year=2021, month=1, day=1, hour=0, minute=0, sec=0)
        self.assertFalse(filter_end_date.condition(record))
        filter_end_date = FilterByEndDate(year=2021, month=1, day=31, hour=23, minute=59, sec=59)
        self.assertFalse(filter_end_date.condition(record))

        
class TestFilterByStartDate(unittest.TestCase):
    
    def test_condition_false(self):
        record = {}
        record[TC.TIMESTAMP] = "2021-01-01 00:00:00"
        filter_end_date = FilterByStartDate(year=2021, month=2, day=1, hour=0, minute=0, sec=0)
        self.assertFalse(filter_end_date.condition(record))
        filter_end_date = FilterByStartDate(year=2021, month=1, day=1, hour=0, minute=0, sec=1)
        self.assertFalse(filter_end_date.condition(record))
    
    def test_condition_true(self):
        record = {}
        record[TC.TIMESTAMP] = "2021-02-01 00:00:00"
        filter_end_date = FilterByStartDate(year=2021, month=1, day=1, hour=0, minute=0, sec=0)
        self.assertTrue(filter_end_date.condition(record))
        filter_end_date = FilterByStartDate(year=2021, month=1, day=31, hour=23, minute=59, sec=59)
        self.assertTrue(filter_end_date.condition(record))

class TestFilterByPeriodicity(unittest.TestCase):
    
    def test_condition_true(self):
        record = {TC.PERIOD: 10}

        for v in Variability:
            record[TC.VARIABILITY] = v
            filter_periodicity = FilterByPeriodicity(v)
            self.assertTrue(filter_periodicity.condition(record))

class TestFilterByNorad(unittest.TestCase):
    
    def test_condition_true(self):
        record = {TC.NORAD_ID: 1}

        indices = [random.randint(0,100) for i in range(10)]
        indices.append(1)
        self.assertTrue(FilterByNorad(indices).condition(record))

    def test_condition_false(self):
        record = {TC.NORAD_ID: 1}

        indices = [random.randint(2,100) for i in range(10)]
        self.assertFalse(FilterByNorad(indices).condition(record))

