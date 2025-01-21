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
    """
    Abstract base class for computing statistics on records.

    Attributes
    ----------
    NAME : str
        The name of the statistic computation class.

    Methods
    -------
    compute(record: dict)
        Abstract method to compute statistics on a given record.
    __call__(record: dict)
        Updates the record with computed statistics and returns it.
    
    Can be used to create custom statistics 

    >>> class MyStat(ComputeStat):
    ...     def __init__(self, param):
    ...         self.param = param
    ...     def compute(self, record: dict):
    ...         return {"MyStat": self.param}
    ...
    >>> MyStat(42)({"a": 1})
    [{'a': 1, 'MyStat': 42}]
    """
    NAME = "ComputeStat"
    
    @abstractmethod
    def compute(self, record: dict):
        """
        Computes statistics on a given record.

        Parameters
        ----------
        record : dict
            The record to compute statistics on.

        Returns
        -------
        dict
            The updated record with computed statistics.
        """
        pass

    def __call__(self, record: dict):
        """
        Updates the record with computed statistics and returns it.

        Parameters
        ----------
        record : dict
            The record to update.

        Returns
        -------
        list of dict
            The updated record wrapped in a list.
        """
        record.update(self.compute(record))
        return [record]

class Amplitude(ComputeStat):
    """
    Computes the amplitude as difference between maximal and minimal value
    of standartized magnitude.

    Methods
    -------
    compute(record: dict)
        Computes the amplitude of the record.
    """

    def compute(self, record: dict):
        """
        Implementation of the amplitude computation.

        Returns
        -------
        dict
            The updated record with the computed amplitude in the "Amplitude" key.
        """
        ok = record[TC.MAG] != 0
        amp = np.max(record[TC.MAG][ok]) - np.min(record[TC.MAG][ok])
        return {"Amplitude": amp}

class MediumTime(ComputeStat):
    """
    Computes the medium time of a given record.

    Methods
    -------
    compute(record: dict)
        Computes the medium time of the record and saves 
        to "MediumTime" key.
    """

    def compute(self, record: dict):
        start = datetime_to_sec(record[TC.TIMESTAMP])
        return {"MediumTime": start + np.mean(record[TC.TIME])}


class MediumPhase(ComputeStat):
    """
    Computes the medium phase of a given record.

    Methods
    -------
    compute(record: dict)
        Computes the medium time of the record and saves 
        to "MediumPhase" key.
    """

    def compute(self, record: dict):
        return {"MediumPhase": np.mean(record[TC.PHASE])}

class FourierSeries(ComputeStat):
    """
    Computes the Fourier series coefficients, amplitude, and covariance of a given record magnitude.

    Attributes
    ----------
    COEFS : str
        Key for storing Fourier coefficients.
    AMPLITUDE : str
        Key for storing Fourier amplitude.
    COVARIANCE : str
        Key for storing Fourier covariance.

    Methods
    -------
    __init__(order, fs=True, covariance=True, amplitude=True)
        Initializes the FourierSeries with the given parameters.
    compute(record: dict)
        Computes the Fourier series coefficients, amplitude, and covariance of the record.
    """

    COEFS = "FourierCoefs"
    AMPLITUDE = "FourierAmplitude"
    COVARIANCE = "FourierCovariance"

    def __init__(self, order, fs=True, covariance=True, amplitude=True):
        """
        Initializes the FourierSeries with the given parameters.

        Parameters
        ----------
        order : int
            The order of the Fourier series.
        fs : bool, optional
            Whether to compute and store Fourier coefficients (default is True).
        covariance : bool, optional
            Whether to compute and store Fourier covariance (default is True).
        amplitude : bool, optional
            Whether to compute and store Fourier amplitude (default is True).
        """
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
    """
    Computes the Continuous Wavelet Transform (CWT) of a given record using the PyWavelet package.

    Attributes
    ----------
    NAME : str
        The name of the statistic computation class.
    wavelet : str
        The type of wavelet to use for the CWT.
    length : int
        The length of the signal to transform.
    scales : array-like
        The number of scales to use for the CWT.

    Methods
    -------
    __init__(wavelet, length, scales)
        Initializes the ContinousWaveletTransform with the given parameters.
    compute(record: dict)
        Computes the Continuous Wavelet Transform of the record.
    """

    NAME = "CWT"

    def __init__(self, wavelet, length, scales):
        """
        Initializes the ContinousWaveletTransform with the given parameters.

        Parameters
        ----------
        wavelet : str
            The type of wavelet to use for the CWT.
        length : int
            The length of the signal to transform.
        scales : array-like
            The scales to use for the CWT.
        """
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
        return {self.NAME: coef.reshape(-1)}