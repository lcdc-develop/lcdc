from abc import ABC, abstractmethod

import numpy as np
import pywt

from .utils import (
    Track,
    fourier_series_fit,
    get_fourier_series,
    datetime_to_sec,
    track_to_grid
)


class ComputeStat(ABC):
    NAME = "ComputeStat"
    
    @abstractmethod
    def compute(self, track: Track) -> Track:
        pass

    def __call__(self, track: Track) -> Track:
        track.stats.update(self.compute(track))

class Amplitude(ComputeStat):

    def compute(self, track: Track) -> Track:
        ok = track.data[:, 1] != 0
        amp = np.max(track.data[ok][:, 1]) - np.min(track.data[ok][:, 1])
        return {"Amplitude": amp}

class MediumTime(ComputeStat):

    def compute(self, track: Track) -> Track:
        start = datetime_to_sec(track.timestamp)
        return {"MediumTime": start + np.mean(track.data[:, 0])}


class MediumPhase(ComputeStat):

    def compute(self, track: Track) -> Track:
        return {"MediumPhase": np.mean(track.data[:, 2])}

class FourierSeries(ComputeStat):
    COEFS = "FourierCoefs"
    AMPLITUDE = "FourierAmplitude"

    def __init__(self, order, fs=True, amplitude=True):
        self.order = order
        self.fs = fs
        self.amplitude = amplitude
    
    def compute(self, track: Track) -> Track:
        period = track.period if track.period != 0 else track.data[-1,0]
        coefs, _ = fourier_series_fit(self.order, track.data, period)

        t = np.linspace(0, track.data[-1,0], 1000)
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
    
    def compute(self, track: Track) -> Track:
        frequency = 1 / self.step
        num = (self.length // self.step) + 1
        if len(lc) != num:
            lc = track_to_grid(track.data, frequency)[:,1]
            if FourierSeries.COEFS in track.stats:
                coefs = track.stats[FourierSeries.COEFS]
            else:
                period = track.period if track.period != 0 else track.data[-1,0]
                coefs, _ = fourier_series_fit(self.order, track.data, period)

        t = np.linspace(0, track.data[-1,0], num, endpoint=True)
        reconstructed = get_fourier_series(self.order, period)(t, *coefs)
        is_zero = lc == 0 
        lc[is_zero] = reconstructed[is_zero]

        scales = np.arange(1, self.scales+1)
        coef, _ = pywt.cwt(lc, scales, self.wavelet)
        return {self.NAME: coef}