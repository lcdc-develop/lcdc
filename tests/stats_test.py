import unittest
import numpy as np
from datetime import datetime
from lcdc.stats import Amplitude, MediumTime, MediumPhase, FourierSeries, ContinousWaveletTransform
from lcdc import Track
from lcdc.utils import datetime_to_sec

class TestStats(unittest.TestCase):

    def setUp(self):
        self.track = Track(0, 0, 0, 0, 3)
        self.track.data = np.ones((100, 5))
        self.track.data[:, 0] = np.arange(100) / 100
        self.track.period = 1
        self.amp = 2
        self.track.data[:, 1] = self.amp/2 * np.sin(self.track.data[:, 0] * 2 * np.pi)
        self.track.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def test_amplitude(self):
        amplitude = Amplitude()
        result = amplitude.compute(self.track)
        self.assertAlmostEqual(result["Amplitude"], self.amp, places=5)

    def test_medium_time(self):
        medium_time = MediumTime()
        result = medium_time.compute(self.track)
        start = datetime_to_sec(*self.track.timestamp.split(" "))
        expected = start + np.mean(self.track.data[:, 0])
        self.assertAlmostEqual(result["MediumTime"], expected, places=5)

    def test_medium_phase(self):
        medium_phase = MediumPhase()
        result = medium_phase.compute(self.track)
        self.assertAlmostEqual(result["MediumPhase"], np.mean(self.track.data[:, 2]), places=5)

    def test_fourier_series(self):
        order = 5
        fourier_series = FourierSeries(order)
        result = fourier_series.compute(self.track)
        self.assertIn(FourierSeries.COEFS, result)
        self.assertIn(FourierSeries.AMPLITUDE, result)

    def test_continous_wavelet_transform(self):
        wavelet = 'morl'
        step = 1
        length = 100
        scales = 5
        cwt = ContinousWaveletTransform(wavelet, step, length, scales)
        result = cwt.compute(self.track)
        self.assertIn(ContinousWaveletTransform.NAME, result)
        self.assertEqual(result[ContinousWaveletTransform.NAME].shape, (scales, length))

if __name__ == "__main__":
    unittest.main()