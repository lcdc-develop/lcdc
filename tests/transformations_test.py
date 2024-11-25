import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.utils import RSO, Track
from lcdc.preprocessing import (
    Fold
)

class FoldTest(unittest.TestCase):

    def test_fold(self):
        t =  Track(0,0,0,0,5,0,-1)
        t.data = np.ones((5,3))
        t.data[:,0] = np.array([0,1,4,5,6])
        t.data[:,1] = np.arange(5)
        t.period = 3

        r = Fold()(t, MagicMock())
        print(r[0].data[:,0] == (np.array([0,0,1,1,2]))/3)

        self.assertAlmostEqual(np.abs(r[0].data[:,0] - (np.array([0,0,1,1,2]))/3).sum(), 0)


