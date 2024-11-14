from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json
from enum import IntEnum, StrEnum
from datetime import datetime
import math

import numpy as np
from scipy import optimize


NANOSEC = 10 ** 9
TENTH_OF_SECOND =  NANOSEC // 10

class StrVars(StrEnum):
    FOURIER_COEFS = 'fourier_coefs'

class Variability(StrEnum):
    APERIODIC = 'aperiodic'
    PERIODIC = 'periodic'
    NONVARIABLE = 'non-variable'

    @staticmethod
    def from_int(n):
        match n:
            case 0:
                return Variability.NONVARIABLE
            case 1:
                return Variability.APERIODIC
            case 2:
                return Variability.PERIODIC
            case _:
                raise ValueError(f"Unknown variability type: {n}")
                

class DataType(StrEnum):
    TIME = 'time'
    MAG = 'mag'
    PHASE = 'phase'
    DISTANCE = 'distance'
    FILTER = 'filter'

ALL_TYPES = [DataType.TIME, DataType.MAG, DataType.PHASE, DataType.DISTANCE, DataType.FILTER]
TYPES_INDICES = {t: i for i, t in enumerate(ALL_TYPES)}


class Filter(IntEnum):
    UNKNOWN = int('00000',2) # 0
    CLEAR   = int('00001',2) # 1
    POL     = int('00010',2) # 2
    V       = int('00100',2) # 4
    R       = int('01000',2) # 8
    B       = int('10000',2) # 16

    @staticmethod
    def str_to_int(s):
        r = 0
        for n in ["Unknown", "Clear", "Pol", "V", "R", "B"]:
            if n in s:
                r = r | Filter[n.upper()].value
        return r
    
    @staticmethod
    def from_int(n):
        res = set()
        for a in Filter:
            if a & n == a:
                res.add(Filter(a))
        return res

@dataclass
class RSO:
    mmt_id : int
    norad_id : int
    name : str
    country : str
    variability : Variability

TRACK_DATA_DIM = 5

@dataclass
class Track:
    id: int
    norad_id: int
    timestamp: int
    mjd: float
    period: int
    start_idx: int = 0
    end_idx: int = -1
    amplitude: float = 0.
    stats: dict = field(default_factory=dict)
    data: np.ndarray = None

    
    # TODO: Maybe differente storing format?? One CSV per Object
    def load_data_from_file(self, dir):
        data = np.loadtxt(f"{dir}/{self.norad_id}/{self.id}.csv", delimiter=',', skiprows=1)
        data = np.atleast_2d(data)
        self.data = data[self.start_idx:None if self.end_idx == -1 else (self.end_idx + 1)]
    
    def load_data(self, data):
        self.data = data[self.start_idx:None if self.end_idx == -1 else (self.end_idx + 1)].copy()

    def unload_data(self):
        del self.data
        self.data = None    

    def to_grid(self, step, size=None):
        lc = self.data

        start = lc[0,0]
        end = lc[-1,0]

        actual_size = np.round((end-start)/step).astype(int)  + 1

        if size is not None:
            if size < actual_size:
                raise ValueError(f"Size must be greater than size of the track: Size {size}, actual size {actual_size}")
            
        else:
            size = actual_size
        

        y = np.zeros((size,TRACK_DATA_DIM))

        counts = np.zeros(size)
        for t, m, p, d, f in lc:
            idx = np.round((t-start)/step).astype(int)
            y[idx, [1,2,3]] += [m, p, d]
            y[idx, 4] = int(y[idx, 4]) | int(f)
            counts[idx] += 1

        ok = counts > 0
        y[ok, 1:4] /= counts[ok, None]
        y[:,0] = np.arange(size) * step + start

        self.data = y
    
    def _gaps_indices(self, max_length=None):
        time_diff = self.data[1:,0] - self.data[:-1,0]
        split_indices, = np.where(time_diff > self.period)

        if max_length is not None:  # connect parts if sum of lengths is less than max_length
            beginnings = self.data[np.concatenate(([0], split_indices+1)),0]
            endings = self.data[np.concatenate((split_indices, [len(self.data)-1])),0]
            part_dist = beginnings[1:] - endings[:-1] 
            part_len = endings - beginnings
            start, end = 0,0
            length = part_len[end]
            split = []

            while end < len(part_len):
                if start == end:
                    if length >= max_length:
                        split.append(end)
                        start += 1
                        length = part_len[start]
                    end += 1

                else:
                    length += part_dist[end-1] + part_len[end]

                    if length >= max_length:
                        split.append(end-1)
                        start = end
                        length = part_len[end]
                    else:
                        end += 1


            split_indices = split_indices[split]
        
        return list(split_indices + 1)

    def split(self, indices):
        start = 0
        parts = []
        ends = indices + [len(self.data)]
        for i, arr in enumerate(np.split(self.data, indices)):
            t = Track(**self.__dict__)
            t.data = arr
            t.start_idx = start + self.start_idx
            t.end_idx = ends[i] + self.start_idx - 1
            parts.append(t)
            
        return parts

    def split_by_gaps(self, max_length=None):

        split_indices = self._gaps_indices(max_length)
        return self.split(split_indices)
    
    def _size_indices(self, max_length, uniform=False):

        split_indices = []
        length = self.data[-1,0] - self.data[0,0] + 1
        if length > max_length:
            if uniform:
                max_length = (length / math.ceil(length / max_length))

            i = 0
            while i < len(self.data):
                start_idx = i
                t_start = self.data[start_idx,0]

                while i < len(self.data) and self.data[i,0] - t_start < max_length:
                    i += 1

                if i < len(self.data):
                    split_indices.append(i)

        return split_indices

    def split_by_size(self, max_length, uniform=False):
        split_indices = self._size_indices(max_length, uniform)
        return self.split(split_indices)
    
    def length(self, step=None):

        if step is None:
            count = np.sum(self.data[:,1] != 0)
            return count
        
        start = self.data[0,0]
        indices = set()
        for t, _, _ in self.data:
            idx = np.round((t-start)/step).astype(int)
            indices.add(idx)

        return len(indices)
            
    def fold(self, k=100):
        period = self.period

        folded = np.zeros(k)
        t = self.data[:,0]
        phases = t / period
        indices = np.round((phases - np.floor(phases)) * k).astype(int)
        for idx in range(k):
            x, = np.where(indices == idx)
            if len(x) > 0:
                folded[idx] = self.data[x,1].mean()

        return folded

    def compute_amplitude(self):
        coefs = self.stats.get(StrVars.FOURIER_COEFS, None)
        if coefs is None:
            coefs, _ = self.get_fourier_coeficients(8)
            self.stats[StrVars.FOURIER_COEFS] = coefs
        
        order = len(coefs) // 2
        reconstructed = np.array([self._fourier(order)(x, *coefs) for x in self.data[:,0]])
        amplitude = np.max(reconstructed) - np.min(reconstructed)
        self.amplitude = amplitude
    
    def _fourier(self, k, period=1):
        def fourier(x, *coefs): 
            pi2 = 2*np.pi
            y = coefs[0]
            for i in range(1, k+1,2):
                a = coefs[2*i-1]
                b = coefs[2*i]
                y += a * np.cos((i-1)*x/period*pi2) + b * np.sin((i-1)*x/period*pi2)
            return y
        return fourier
    
    def get_fourier_coeficients(self,k):
        """Computes fourier coeficients for the track

        Args:
            k (int): Order of the fourier series

        Returns:
            Tuple[1D Array, 2D Array]: (Coeficients, Covariance matrix) with 2k+1 coeficients
        """
        period = self.period if self.period != 0 else self.data[-1,0]
        non_zero = self.data[:,1] != 0
        return optimize.curve_fit(self._fourier(k, period), 
                                                xdata=self.data[:,0][non_zero],
                                                ydata=self.data[:,1][non_zero],
                                                p0=np.ones(2*k+1),
                                                absolute_sigma=False,
                                                method="lm",
                                                maxfev=10000)


def datetime_to_nanosec(date, time):
    sec = datetime.strptime(f'{date} {time}', f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}").timestamp()
    return int(sec * NANOSEC) 

def datetime_to_sec(date, time):
    return datetime.strptime(f'{date} {time}', f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}").timestamp()

def sec_to_datetime(sec):
    return datetime.fromtimestamp(sec).strftime("%Y-%m-%d %H:%M:%S.%f")

def load_rsos_from_csv(filename, header=True) -> List[RSO]:
    with open(filename, 'r') as f:
        rsos = []
        for line in f.readlines()[1 if header else 0:]:
            mmt_id, norad_id, name, country, var = line.strip().split(',')
            rsos.append( RSO(int(mmt_id), int(norad_id), name, country, var))
        return rsos

def load_tracks_from_csv(filename, header=True) -> List[Track]:
    #TODO: Tracks will be stored in single CSV per object
    with open(filename, 'r') as f:
        tracks = []
        for line in f.readlines()[1 if header else 0:]:
            s = line.strip().split(',')
            id, norad_id, time, mjd, period = s[:5]
            s_idx = 0 if len(s) < 6 else s[5]
            e_idx = -1 if len(s) < 6 else s[6]

            tracks.append( Track(int(id),int(norad_id),time,float(mjd),float(period),int(s_idx),int(e_idx)))
        return tracks