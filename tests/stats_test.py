import unittest
import numpy as np
from datetime import datetime
from lcdc.stats import Amplitude, MediumTime, MediumPhase, FourierSeries, ContinousWaveletTransform
from lcdc.utils import datetime_to_sec
from lcdc.vars import TableCols as TC, DATA_COLS

class TestStats(unittest.TestCase):

    def setUp(self):
        record = {TC.PERIOD: 1}
        for c in DATA_COLS:
            record[c] = np.ones(100)
        record[TC.TIME] = np.arange(100) / 100
        record[TC.MAG] = np.sin(record[TC.TIME] * 2 * np.pi)
        record[TC.TIMESTAMP] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.record = record


    def test_amplitude(self):
        amplitude = Amplitude()
        result = amplitude.compute(self.record)
        self.assertAlmostEqual(result["Amplitude"], 2, places=5)

    def test_medium_time(self):
        medium_time = MediumTime()
        result = medium_time.compute(self.record)
        start = datetime_to_sec(self.record[TC.TIMESTAMP])
        expected = start + np.mean(self.record[TC.TIME])
        self.assertAlmostEqual(result["MediumTime"], expected, places=5)

    def test_medium_phase(self):
        medium_phase = MediumPhase()
        result = medium_phase.compute(self.record)
        self.assertAlmostEqual(result["MediumPhase"], np.mean(self.record[TC.PHASE]), places=5)

    def test_fourier_series(self):
        order = 5
        fourier_series = FourierSeries(order)
        result = fourier_series.compute(self.record)
        self.assertIn(FourierSeries.COEFS, result)
        self.assertIn(FourierSeries.AMPLITUDE, result)

    def test_continous_wavelet_transform(self):
        wavelet = 'morl'
        length = 50
        scales = 5
        cwt = ContinousWaveletTransform(wavelet, length, scales)
        result = cwt.compute(self.record)
        self.assertIn(ContinousWaveletTransform.NAME, result)
        self.assertEqual(result[ContinousWaveletTransform.NAME].shape, (scales, length))

if __name__ == "__main__":
    unittest.main()