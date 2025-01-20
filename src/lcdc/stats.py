from abc import ABC, abstractmethod

import numpy as np
import pywt

from .vars import TableCols as TC
from .utils import (
    fourier_series_fit,
    get_fourier_series,
    datetime_to_sec,
    to_grid
)


class ComputeStat(ABC):
    NAME = "ComputeStat"
    
    @abstractmethod
    def compute(self, record: dict):
        pass

    def __call__(self, record: dict):
        record.update(self.compute(record))
        return [record]

class Amplitude(ComputeStat):

    def compute(self, record: dict):
        ok = record[TC.MAG] != 0
        amp = np.max(record[TC.MAG][ok]) - np.min(record[TC.MAG][ok])
        return {"Amplitude": amp}

class MediumTime(ComputeStat):

    def compute(self, record: dict):
        start = datetime_to_sec(record[TC.TIMESTAMP])
        return {"MediumTime": start + np.mean(record[TC.TIME])}


class MediumPhase(ComputeStat):

    def compute(self, record: dict):
        return {"MediumPhase": np.mean(record[TC.PHASE])}

class FourierSeries(ComputeStat):
    COEFS = "FourierCoefs"
    AMPLITUDE = "FourierAmplitude"
    COVARIANCE = "FourierCovariance"

    def __init__(self, order, fs=True, covariance=True, amplitude=True):
        self.order = order
        self.fs = fs
        self.amplitude = amplitude
        self.covar = covariance
    
    def compute(self, record: dict):
        period = record[TC.PERIOD] if record[TC.PERIOD] != 0 else record[TC.TIME][-1]
        coefs, covar = fourier_series_fit(self.order, record, period)

        t = np.linspace(0, record[TC.TIME][-1], 1000)
        reconstructed = get_fourier_series(self.order, period)(t, *coefs)
        amplitude = np.max(reconstructed) - np.min(reconstructed)
        res = {}
        if self.fs:
            res[self.COEFS] = coefs
        if self.amplitude:
            res[self.AMPLITUDE] = amplitude
        if self.covar:
            res[self.COVARIANCE] = covar.reshape(-1)
        return res
    

class ContinousWaveletTransform(ComputeStat):
    NAME = "CWT"

    def __init__(self, wavelet, length, scales):
        self.wavelet = wavelet
        self.length = length
        self.scales = scales
    
    def compute(self, record: dict):
        step = record[TC.TIME][-1] / (self.length)
        frequency = (self.length-1) / (record[TC.TIME][-1])
        print(frequency, step, self.length, step * self.length, record[TC.TIME][-1])
        num = self.length 
        period = record[TC.PERIOD] if record[TC.PERIOD] != 0 else record[TC.TIME][-1]

        if FourierSeries.COEFS in record:
            coefs = record[FourierSeries.COEFS]
        else:
            coefs, _ = fourier_series_fit(8, record, period)

        if len(record[TC.TIME]) != num:
            record = to_grid(record, frequency)

        t = np.linspace(0, record[TC.TIME][-1], num, endpoint=True)
        reconstructed = get_fourier_series(8, period)(t, *coefs)
        lc = record[TC.MAG].copy()
        print(record[TC.TIME].shape, lc.shape)
        is_zero = lc == 0 
        lc[is_zero] = reconstructed[is_zero]

        scales = np.arange(1, self.scales+1)
        coef, _ = pywt.cwt(lc, scales, self.wavelet)
        print(coef.shape)
        return {self.NAME: coef}