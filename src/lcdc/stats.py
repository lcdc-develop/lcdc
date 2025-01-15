from abc import ABC, abstractmethod

import numpy as np
import pywt

from .vars import TableCols as TC
from .utils import (
    fourier_series_fit,
    get_fourier_series,
    datetime_to_sec,
    track_to_grid
)


class ComputeStat(ABC):
    NAME = "ComputeStat"
    
    @abstractmethod
    def compute(self, record: dict):
        pass

    def __call__(self, record: dict):
        record.update(self.compute(record))

class Amplitude(ComputeStat):

    def compute(self, record: dict):
        ok = record[TC.DATA][:, 1] != 0
        amp = np.max(record[TC.DATA][ok][:, 1]) - np.min(record[TC.DATA][ok][:, 1])
        return {"Amplitude": amp}

class MediumTime(ComputeStat):

    def compute(self, record: dict):
        start = datetime_to_sec(record[TC.TIMESTAMP])
        return {"MediumTime": start + np.mean(record[TC.DATA][:, 0])}


class MediumPhase(ComputeStat):

    def compute(self, record: dict):
        return {"MediumPhase": np.mean(record[TC.DATA][:, 2])}

class FourierSeries(ComputeStat):
    COEFS = "FourierCoefs"
    AMPLITUDE = "FourierAmplitude"

    def __init__(self, order, fs=True, amplitude=True):
        self.order = order
        self.fs = fs
        self.amplitude = amplitude
    
    def compute(self, record: dict):
        period = record[TC.PERIOD] if record[TC.PERIOD] != 0 else record[TC.DATA][-1,0]
        coefs, _ = fourier_series_fit(self.order, record[TC.DATA], period)

        t = np.linspace(0, record[TC.DATA][-1,0], 1000)
        reconstructed = get_fourier_series(self.order, period)(t, *coefs)
        amplitude = np.max(reconstructed) - np.min(reconstructed)
        res = {}
        if self.fs:
            res[self.COEFS] = coefs
        if self.amplitude:
            res[self.AMPLITUDE] = amplitude
        return res
    

class ContinousWaveletTransform(ComputeStat):
    NAME = "CWT"

    def __init__(self, wavelet, step, length, scales):
        self.wavelet = wavelet
        self.step = step
        self.length = length
        self.scales = scales
    
    def compute(self, record: dict):
        frequency = 1 / self.step
        num = (self.length // self.step) + 1
        if len(lc) != num:
            lc = track_to_grid(record[TC.DATA], frequency)[:,1]
            if FourierSeries.COEFS in record[TC.STATS]:
                coefs = record[TC.STATS][FourierSeries.COEFS]
            else:
                period = record[TC.PERIOD] if record[TC.PERIOD] != 0 else record[TC.DATA][-1,0]
                coefs, _ = fourier_series_fit(self.order, record[TC.DATA], period)

        t = np.linspace(0, record[TC.DATA][-1,0], num, endpoint=True)
        reconstructed = get_fourier_series(self.order, period)(t, *coefs)
        is_zero = lc == 0 
        lc[is_zero] = reconstructed[is_zero]

        scales = np.arange(1, self.scales+1)
        coef, _ = pywt.cwt(lc, scales, self.wavelet)
        return {self.NAME: coef}