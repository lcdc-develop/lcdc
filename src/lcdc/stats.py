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
        amp = np.max(track.data[:, 1]) - np.min(track.data[:, 1])
        return {"Amplitude": amp}

class MediumTime(ComputeStat):

    def compute(self, track: Track) -> Track:
        start = datetime_to_sec(*track.timestamp.split(" "))
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
        num = self.length // self.step
        lc = track_to_grid(track.data, self.length, num)
        scales = np.arange(1, self.scales+1)
        coef, _ = pywt.cwt(lc[:,1] ,scales,self.wavelet)
        return {self.NAME: coef}

if __name__ == "__main__":
    
    track = Track(0,0,0,0,3)
    track.data = np.ones((100, 5))
    track.data[:, 0] = np.arange(100) / 100 
    track.period = 1
    amp = 2
    track.data[:, 1] = amp* np.sin(track.data[:, 0] * 2 * np.pi)
    object = None