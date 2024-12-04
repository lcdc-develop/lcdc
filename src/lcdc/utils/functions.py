from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from datetime import datetime
from typing import List

from ..vars import *
from .rso import RSO
from .track import Track


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
    with open(filename, 'r') as f:
        tracks = []
        for line in f.readlines()[1 if header else 0:]:
            try:
                s = line.strip().split(',')
                id, norad_id, time, mjd, period = s[:5]
                s_idx = 0 if len(s) < 6 else s[5]
                e_idx = -1 if len(s) < 6 else s[6]

                tracks.append( Track(int(id),int(norad_id),time,float(mjd),float(period),int(s_idx),int(e_idx)))
            except:
                pass
        return tracks
    
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

def fourier_series_fit(order, data, period):
    non_zero = data[:,1] != 0
    return optimize.curve_fit(get_fourier_series(order, period), 
                                            xdata=data[:,0][non_zero],
                                            ydata=data[:,1][non_zero],
                                            p0=np.ones(2*order+1),
                                            absolute_sigma=False,
                                            method="lm",
                                            maxfev=10000)

def plot_track(track, mag=True, phase=False, dist=False, fourier=False, mag_phase=False):

    n_plots = sum([mag, phase, dist, fourier, mag_phase])
    height_ratios = [r for r, b in zip([6,2,4,4,4], [mag, fourier,mag_phase, phase, dist]) if b]

    fig, axs = plt.subplots(n_plots,1, gridspec_kw={"height_ratios": height_ratios}, figsize=(10, 2*n_plots))

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
        fs = get_fourier_series(8, period)
        coefs = fourier_series_fit(8, track.data, period)[0]
        fit = fs(t, *coefs)
        axs[0].plot(t, fit, c='red', label="Fourier series")

        # plot residuals
        reconstructed = fs(track.data[:,0], *coefs)
        residuals = track.data[:,1] - reconstructed
        rms = np.sqrt(np.mean(residuals**2))
        axs[idx].set_title(f"Residuals: (RMS: {rms:.2f}, max: {residuals.max():.2f}) ")
        axs[idx].scatter(track.data[:, 0], residuals, s=1, c='red')
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

def fold_track(data: np.ndarray, period: float) -> np.ndarray:
    # if len(data) == 113:
        # print(data[:,0], period)
    data = data.copy()
    time = np.modf(data[:,0] / period)[0]
    indices = np.argsort(time)
    data = data[indices]
    data[:,0] = time[indices]

    return data

def track_to_grid(data, length, num):

    step = length / num

    bin_indices = np.modf(data[:,0]/ step)[1]
    # sort bin_indices 
    indices = np.argsort(bin_indices)
    bin_indices = bin_indices[indices]
    data = data[indices]

    # groupby bin_indices
    bin_idx, split_indices = np.unique(bin_indices, return_index=True)
    bin_values = np.split(data[:,1:], split_indices[1:], axis=0)
    # if len(bin_idx) == 2:
        # print(bin_idx, step, len(data))
        # print(bin_indices, data[-1])

    grid = np.zeros((num, data.shape[1]))
    grid[:,0] = np.arange(num) * step

    for idx, values in zip(bin_idx, bin_values):
        idx = int(idx)
        grid[idx, 1:-1] = np.mean(values[:,:-1], axis=0)
        # XOR the Filter column
        grid[idx, -1] = reduce(operator.xor, values[:,-1].astype(np.int16))


    return grid
        