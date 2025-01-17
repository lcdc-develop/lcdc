from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from datetime import datetime
from typing import List

from ..vars import *
from ..vars import TableCols as TC, DATA_COLS


def datetime_to_nanosec(timestamp):
    date, time = timestamp.split(' ')
    sec = datetime.strptime(f'{date} {time}', f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}").timestamp()
    return int(sec * NANOSEC) 

def datetime_to_sec(timestamp):
    date, time = timestamp.split(' ')
    return datetime.strptime(f'{date} {time}', f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}").timestamp()

def sec_to_datetime(sec):
    return datetime.fromtimestamp(sec).strftime("%Y-%m-%d %H:%M:%S.%f")

def get_fourier_series(order, period=1):
    def fun(x, *coefs): 
        pi2 = 2*np.pi
        y = coefs[0]
        for i in range(1, order+1):
            a = coefs[2*i-1]
            b = coefs[2*i]
            y += a * np.cos(i*x/period*pi2) + b * np.sin(i*x/period*pi2)
        return y
    return fun

def fourier_series_fit(order, record, period):
    non_zero = record[TC.MAG] != 0
    return optimize.curve_fit(get_fourier_series(order, period), 
                                            xdata=record[TC.TIME][non_zero],
                                            ydata=record[TC.MAG][non_zero],
                                            p0=np.ones(2*order+1),
                                            absolute_sigma=False,
                                            method="lm",
                                            maxfev=10000)

def plot_track(track, mag=True, phase=False, dist=False, fourier=False, mag_phase=False):

    n_plots = sum([mag, phase, dist, fourier, mag_phase])
    height_ratios = [r for r, b in zip([6,2,4,4,4], [mag, fourier,mag_phase, phase, dist]) if b]

    fig, axs = plt.subplots(n_plots,1, gridspec_kw={"height_ratios": height_ratios}, figsize=(10, 2*n_plots))
    if n_plots == 1:
        axs = [axs]

    idx = 0
    axs[-1].set_xlabel("Time [sec]")
    if mag:
        axs[idx].scatter(track.data[:, 0], track.data[:,1], s=1, c='blue')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Track {track.id}, NORAD: {track.norad_id}")
        axs[idx].set_ylabel("Magnitude")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1

    if fourier:

        t = np.linspace(track.data[0,0], track.data[-1,0], 1000)
        period = track.period if track.period != 0 else track.data[-1,0]
        k = 8
        if "FourierCoefs" in track.stats:
            coefs = track.stats["FourierCoefs"]
            k = len(coefs) // 2
            fs = get_fourier_series(k, period)
        else:
            fs = get_fourier_series(8, period)
            coefs = fourier_series_fit(8, track.data, period)[0]
        fit = fs(t, *coefs)
        axs[0].plot(t, fit, c='red', label="Fourier series")
        axs[0].legend()

        # plot residuals
        reconstructed = fs(track.data[:,0], *coefs)
        residuals = track.data[:,1] - reconstructed
        ok = track.data[:,1] != 0
        rms = np.sqrt(np.mean(residuals**2))
        axs[idx].set_title(f"Residuals: (RMS: {rms:.2f}, max: {residuals.max():.2f}) ")
        axs[idx].scatter(track.data[:, 0][ok], residuals[ok], s=1, c='red')
        axs[idx].set_ylabel("Δ Magnitude")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1

    if mag_phase:
        axs[idx].set_title(f"Phase vs Magnitude")
        axs[idx].scatter(track.data[:, 2], track.data[:,1], s=1, c='blue')
        axs[idx].invert_yaxis()
        axs[idx].set_ylabel("Mag")
        axs[idx].set_xlabel("Phase")
        idx += 1
    
    if phase:
        axs[idx].scatter(track.data[:, 0], track.data[:,2], s=1, c='green')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Phase")
        axs[idx].set_ylabel("Phase [Deg]")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1
    
    if dist:
        axs[idx].scatter(track.data[:, 0], track.data[:,3], s=1, c='orange')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Discance")
        axs[idx].set_ylabel("Distance [km]")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1
    plt.tight_layout()
    
    return fig, axs

def plot_from_dict(data):
    t = Track(data["id"], data["norad_id"], data["timestamp"], 0, data["period"], data["start_idx"], data["end_idx"])
    
    time = None
    mag = None
    phase = None
    size = t.period
    if DataType.TIME in data:
        time = np.array(data[DataType.TIME])
        size = len(time)
    else:
        for dt in [DataType.MAG, DataType.PHASE, DataType.DISTANCE]:
            if dt in data:
                size = len(data[dt])
                break
        time = np.linspace(0, size, size)
    if DataType.MAG in data:
        mag = np.array(data[DataType.MAG])
    else:
        mag = np.zeros(size)
    if DataType.PHASE in data:
        phase = np.array(data[DataType.PHASE])
    else:
        phase = np.zeros(size)
    if DataType.DISTANCE in data:
        dist = np.array(data[DataType.DISTANCE])
    else:
        dist = np.zeros(size)
    
    t.data = np.stack([time, mag, phase, dist], axis=1)
    if "FourierCoefs" in data:
        t.stats["FourierCoefs"] = data["FourierCoefs"]
        
    return plot_track(t, mag=DataType.MAG in data, phase=DataType.PHASE in data, dist=DataType.DISTANCE in data, fourier="FourierCoefs" in data,
               mag_phase=DataType.MAG in data and DataType.PHASE in data)


def fold(record, period: float):
    record = record.copy()
    time = np.modf(record[TC.TIME] / period)[0]
    indices = np.argsort(time)
    for c in filter(lambda x: x in record, DATA_COLS):
        if c == TC.TIME: continue
        record[c] = record[c][indices]
    record[TC.TIME] = time[indices]
    return record

def to_grid(record, sampling_frequency):
    record = record.copy()
    step = 1/sampling_frequency
    num = int(record[TC.TIME][-1] / step + 1e-5 ) + 1

    bin_indices = np.modf(record[TC.TIME]/ step)[1]

    indices = np.argsort(bin_indices)
    bin_indices = bin_indices[indices]
    for c in filter(lambda x: x in record, DATA_COLS):
        record[c] = record[c][indices]

    # groupby bin_indices
    bin_idx, split_indices = np.unique(bin_indices, return_index=True)
    # bin_values = np.split(data[:,1:], split_indices[1:], axis=0)
    values = {c: record[c].copy() for c in filter(lambda x: x in record, DATA_COLS)}

    ends = list(split_indices[1:])+[num]

    for c in filter(lambda x: x in record, DATA_COLS):
        record[c] = np.zeros(num)
    record[TC.TIME] = np.arange(num) * step

    start = 0
    for idx, end in zip(bin_idx, ends):
        idx = int(idx)
        for c in filter(lambda x: x in record, DATA_COLS):
            if c == TC.TIME: continue
            if c == TC.FILTER:
                record[c][idx] = reduce(operator.xor, values[c][start:end].astype(np.int16),0)
            else:
                record[c][idx] = np.mean(values[c][start:end])
        start = end

    return record
        
