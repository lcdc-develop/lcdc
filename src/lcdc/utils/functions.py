import operator
from datetime import datetime
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from ..vars import DATA_COLS, TableCols as TC


def datetime_to_sec(timestamp):
    date, time = timestamp.split(' ')
    return datetime.strptime(
        f'{date} {time}', f"%Y-%m-%d %H:%M:%S{'.%f' if '.' in time else ''}"
    ).timestamp()


def sec_to_datetime(sec):
    return datetime.fromtimestamp(sec).strftime("%Y-%m-%d %H:%M:%S.%f")

def get_fourier_series(order, period=1):
    def fun(x, *coefs):
        pi2 = 2 * np.pi
        y = coefs[0]
        for i in range(1, order + 1):
            a = coefs[2 * i - 1]
            b = coefs[2 * i]
            y += a * np.cos(i * x / period * pi2) + b * np.sin(i * x / period * pi2)
        return y
    return fun

def fourier_series_fit(order, record, period):
    non_zero = record[TC.MAG] != 0
    return optimize.curve_fit(
        get_fourier_series(order, period),
        xdata=record[TC.TIME][non_zero],
        ydata=record[TC.MAG][non_zero],
        p0=np.ones(2 * order + 1),
        absolute_sigma=False,
        method="lm",
        maxfev=10000,
    )


def plot_track(record):

    mag = TC.MAG in record
    fourier = "FourierCoefs" in record
    mag_phase = TC.MAG in record and TC.PHASE in record
    phase = TC.PHASE in record
    dist = TC.DISTANCE in record

    height_ratios = [
        r for r, b in zip([6, 2, 4, 4, 4], [mag, fourier, mag_phase, phase, dist]) if b
    ]
    n_plots = len(height_ratios)

    fig, axs = plt.subplots(
        n_plots, 1, gridspec_kw={"height_ratios": height_ratios}, figsize=(10, 2 * n_plots)
    )
    if n_plots == 1:
        axs = [axs]

    for c in DATA_COLS:
        if c in record:
            record[c] = np.array(record[c])

    idx = 0
    axs[-1].set_xlabel("Time [sec]")
    if mag:
        axs[idx].scatter(record[TC.TIME], record[TC.MAG], s=1, c='blue')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Track {record[TC.ID]}, NORAD: {record[TC.NORAD_ID]}")
        axs[idx].set_ylabel("Magnitude")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1

    if fourier:

        t = np.linspace(record[TC.TIME][0], record[TC.TIME][-1], 1000)
        period = record[TC.PERIOD] if record[TC.PERIOD] != 0 else record[TC.TIME][-1]
        k = 8
        if "FourierCoefs" in record:
            coefs = record["FourierCoefs"]
            k = len(coefs) // 2
            fs = get_fourier_series(k, period)
        else:
            fs = get_fourier_series(8, period)
            coefs = fourier_series_fit(8, record, period)[0]
        fit = fs(t, *coefs)
        axs[0].plot(t, fit, c='red', label="Fourier series")
        axs[0].legend()

        # plot residuals
        reconstructed = fs(record[TC.TIME], *coefs)
        residuals = record[TC.MAG] - reconstructed
        ok = record[TC.MAG] != 0
        rms = np.sqrt(np.mean(residuals**2))
        axs[idx].set_title(f"Residuals: (RMS: {rms:.2f}, max: {residuals.max():.2f}) ")
        axs[idx].scatter(record[TC.TIME][ok], residuals[ok], s=1, c='red')
        axs[idx].set_ylabel("Δ Magnitude")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1

    if mag_phase:
        axs[idx].set_title(f"Phase vs Magnitude")
        axs[idx].scatter(record[TC.PHASE], record[TC.MAG], s=1, c='blue')
        axs[idx].invert_yaxis()
        axs[idx].set_ylabel("Mag")
        axs[idx].set_xlabel("Phase")
        idx += 1

    if phase:
        axs[idx].scatter(record[TC.TIME], record[TC.PHASE], s=1, c='green')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Phase")
        axs[idx].set_ylabel("Phase [Deg]")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1

    if dist:
        axs[idx].scatter(record[TC.TIME], record[TC.DISTANCE], s=1, c='orange')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Discance")
        axs[idx].set_ylabel("Distance [km]")
        axs[idx].set_xlabel("Time [sec]")
        idx += 1
    plt.tight_layout()

    return fig, axs

def fold(record, period: float):
    record = record.copy()
    time = np.modf(record[TC.TIME] / period)[0]
    indices = np.argsort(time)
    for c in filter(lambda x: x in record, DATA_COLS):
        if c == TC.TIME:
            continue
        record[c] = record[c][indices]
    record[TC.TIME] = time[indices]
    return record

def to_grid(record, sampling_frequency):
    record = record.copy()
    step = 1 / sampling_frequency
    num = int(record[TC.TIME][-1] / step + 1e-5) + 1

    bin_indices = np.modf(record[TC.TIME] / step)[1]

    indices = np.argsort(bin_indices)
    bin_indices = bin_indices[indices]
    for c in filter(lambda x: x in record, DATA_COLS):
        record[c] = record[c][indices]

    # groupby bin_indices
    bin_idx, split_indices = np.unique(bin_indices, return_index=True)
    values = {c: record[c].copy() for c in (x for x in DATA_COLS if x in record)}

    ends = list(split_indices[1:]) + [num]

    for c in filter(lambda x: x in record, DATA_COLS):
        record[c] = np.zeros(num)
    record[TC.TIME] = np.arange(num) * step

    start = 0
    for idx, end in zip(bin_idx, ends):
        idx = int(idx)
        for c in filter(lambda x: x in record, DATA_COLS):
            if c == TC.TIME:
                continue
            if c == TC.FILTER:
                record[c][idx] = reduce(operator.xor, values[c][start:end].astype(np.int16), 0)
            else:
                record[c][idx] = np.mean(values[c][start:end])
        start = end

    return record