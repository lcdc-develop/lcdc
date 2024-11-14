import unittest
import numpy as np

from utils import RSO, Track




class TestTrack(unittest.TestCase):

    def _get_track(self):
        return Track(
            id=0,
            norad_id=0,
            timestamp=0,
            mjd=0,
            period=3,
        )

    def test_split(self):
        t = self._get_track()
        t.data = np.random.rand(15,3)
        t.data[:,0] = np.array([0,0,0,10,10,10,10,20,20,20,20,20,20,30,30])

        r = t.split_by_gaps()

        assert len(r) == 4, "Wrong number of splits"
        for i, s in enumerate([3,4,6,2]):
            assert len(r[i].data) == s, f"Wrong length for split {i}"
    
    def test_split_max_length(self):
        t = self._get_track()
        t.data = np.random.rand(15,3)
        t.data[:,0] = np.array([0,0,0,10,11,12,13,23,24,25,26,27,28,38,39])
        t.data[:,1] = np.arange(15)

        r = t.split_by_gaps(max_length=25)
        assert len(r) == 2, f"Wrong number of splits: Expected 2, got {len(r)}"
        for i, s in enumerate([7,8]):
            assert len(r[i].data) == s, f"Wrong length for split {i}: Expected {s}, got {len(r[i].data)}"

        r = t.split_by_gaps(max_length=15)
        # print(r)
        assert len(r) == 3, "Wrong number of splits"
        for i, s in enumerate([7,6,2]):
            assert len(r[i].data) == s, f"Wrong length for split {i}: Expected {s}, got {len(r[i].data)}"
    
    def test_split_by_size(self):
        t = self._get_track()
        t.data = np.random.rand(15,3)
        t.data[:,0] = np.array([0,0,0,10,10,10,10,20,20,20,20,20,20,30,30])
        r = t.split_by_size(9)

        assert len(r) == 4, f"Wrong number of splits: Expected 4, got {len(r)}"
        for i, s in enumerate([3,4,6,2]):
            assert len(r[i].data) == s, f"Wrong length for split {i}: Expected {s}, got {len(r[i].data)}"

        t.data = np.random.rand(20,3)
        t.data[:,0] = np.arange(20)
        r = t.split_by_size(7)

    def test_split_by_size_uniform(self):
        t = self._get_track()
        t.data = np.random.rand(15,3)
        t.data[:,0] = np.arange(15)
        r = t.split_by_size(11, uniform=True)

        assert len(r) == 2, f"Wrong number of splits: Expected 2, got {len(r)}"
        for i, s in enumerate([8,7]):
            assert len(r[i].data) == s, f"Wrong length for split {i}: Expected {s}, got {len(r[i].data)}"
    
    def test_fold(self):
        t = self._get_track()
        t.data = np.ones((5,3))
        t.data[:,0] = np.array([0,1,4,5,6])
        t.period = 3

        r = t.fold(3)
        assert np.sum(r) == 3, f"Erorr: Expected {[1,1,1]}, got {r}"

        r = t.fold(6)
        e = np.array([1,0,1,0,1,0])
        assert np.all(r==e) , f"Erorr: Expected {e}, got {r}"

        t.data[:,0] = np.array([0,1,3,4,6])
        r = t.fold(3)
        e = np.array([1,1,0])
        assert np.all(r==e) , f"Erorr: Expected {e}, got {r}"
    
    def test_to_grid(self):
        t = self._get_track()
        t.data = np.ones((5,5))
        t.data[:,0] = np.array([0,1,4,5,6])

        t.to_grid(1)
        r = t.data
        assert r.shape[0] == 7

        t.data = np.ones((5,5))
        t.data[:,0] = np.array([0,1,4,5,6])
        t.to_grid(1, size=10)
        r = t.data
        assert r.shape[0] == 10
    
    def test_amplitude(self):
        t = self._get_track()
        t.data = np.ones((100, 5))
        t.data[:, 0] = np.arange(100) / 100 
        amp = 3
        t.data[:, 1] = amp* np.sin(t.data[:, 0] * 2 * np.pi)

        eps = 1e-3
        t.compute_amplitude()
        assert  2*amp - eps < t.amplitude < 2*amp + eps, f"Wrong amplitude: Expected {amp}, got {t.amplitude}"


        t.data[:, 1] = amp* np.sin(t.data[:, 0] * 2 * np.pi) + amp* np.sin(t.data[:, 0] * 2 * np.pi - np.pi)
        t.compute_amplitude()
        assert 0 - eps < t.amplitude < 0 + eps, f"Wrong amplitude: Expected 0, got {t.amplitude}"
    
    def test_length(self):
        t = self._get_track()
        t.data = np.ones((10,3))
        t.data[:,0] = np.arange(10)

        l = t.length()
        assert 10 == l, f"Wrong length: Expected 10, got {l}"

        l = t.length(step=2)
        assert 5 == l, f"Wrong length: Expected 5, got {l}"

        t.data[:,0] = np.array([0,1,2,3,4,5,7,8,10,11])
        assert t.length() == 10, f"Wrong length: Expected 10, got {t.length()}"
        assert t.length(step=2) == 6, f"Wrong length: Expected 6, got {t.length(step=2)}"


if __name__ == "__main__":
    unittest.main()