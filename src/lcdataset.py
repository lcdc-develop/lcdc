import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
from utils import DataType, TYPES_INDICES, Track

# HACK: test if stats work correctly

class LCDataset:

    COLUMNS = ["id", "label", "period", "amplitude", "start_idx", "end_idx"]

    def __init__(
        self,
        tracks: Dict[int, Track],
        labels: Dict[int, str],
        data_dir: str,
        compute_mean_std: bool = False,
        name: str = "data",
    ):

        self.norad_to_label = labels
        self.tracks = tracks
        self.label_idx = {c: i for i, c in enumerate(sorted(set(labels.values())))}
        self.data_dir = data_dir
        self.mean_std = compute_mean_std
        self.name = name

    @staticmethod
    def dict_from_file(path):
        files = os.listdir(path)
        csvs = [f for f in files if f.endswith(".csv") and f != "mean_std.csv"]

        assert len(csvs) == 1, f"No dataset csv file found in {path}"

        metadata_csv = csvs[0]
        data = defaultdict(list)
        with open(f"{path}/{metadata_csv}", "r") as f:
            f.readline()
            for line in f.readlines():
                id, label, p, amp, s_idx, e_idx, file = line.strip().split(",")

                data["id"].append(int(id))
                data["label"].append(label)
                data["period"].append(float(p))
                data["amplitude"].append(float(amp))
                data["start_idx"].append(int(s_idx))
                data["end_idx"].append(int(e_idx))

                if os.path.exists(f"{path}/{file}"):
                    with open(f"{path}/{file}", "r") as data_file:
                        columns = data_file.readline().strip().split(",")

                    arr = np.loadtxt(f"{path}/{file}", delimiter=",", skiprows=1)
                    for i, c in enumerate(columns):
                        data[c].append(arr[:, i])

        means, stds = LCDataset._load_mean_std(path)
        LCDataset._load_stats_to_data(path, data)

        return {
            "data": data,
            "mean": means,
            "std": stds,
        }

    @staticmethod
    def _load_stats_to_data(path, data):
        json_file = f"{path}/stats.json"
        if os.path.exists(json_file):
            stats = {}
            with open(json_file, "r") as f:
                stats = json.load(f)
            for id in data["id"]:
                for k in stats[id]:
                    data[k].append(stats[id][k])

    @staticmethod
    def _load_mean_std(path):
        means = {}
        stds = {}
        filename = f"{path}/mean_std.csv"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                f.readline()
                for line in f.readlines():
                    k, m, s = line.split(",")
                    means[k] = float(m)
                    stds[k] = float(s)

        return means, stds

    def to_dict(self, data_types):
        data = defaultdict(list)
        stats = False
        stats_names = []

        for t_id in self.tracks:
            for t in self.tracks[t_id]:
                if not stats and t.stats != {}:
                    stats = True
                    stats_names = list(t.stats.keys())
                if t.data is None:
                    t.load_data_from_file(f"{self.data_dir}/data")
                for td in data_types:
                    data[td].append(t.data[:, TYPES_INDICES[td]])
                data["label"].append(self.norad_to_label[t.norad_id])
                data["period"].append(t.period)
                data["amplitude"].append(t.amplitude)
                data["start_idx"].append(t.start_idx)
                data["end_idx"].append(t.end_idx)
                data["id"].append(t.id)

                # HACK: to ensure there is entry for missing stats
                #       maybe should raise error
                if stats:
                    for k, v in t.stats.items():
                        data[k].append(v)

        means = {dt: 0 for dt in data_types + ["amplitude"]}
        stds = {dt: 1 for dt in data_types + ["amplitude"]}

        if self.mean_std:

            for k in means:
                if isinstance(data[k][0], np.ndarray):
                    x = np.concatenate(data[k], axis=0)
                else:
                    x = np.array(data[k])
                non_zero = x != 0
                means[k] = np.mean(x[non_zero])
                stds[k] = np.std(x[non_zero])

        return {
            "data": data,
            "mean": means,
            "std": stds,
        }

    def to_file(self, path, data_types):

        os.makedirs(path, exist_ok=True)
        self._save_metadata(path)
        for c in self.label_idx:
            os.makedirs(f"{path}/data/{self._remove_backslash(c)}", exist_ok=True)

        save_stats = False

        running_std = {
            dt: RunningMeanStd()
            for dt in data_types
            if dt != DataType.TIME and dt != DataType.FILTER
        }

        for parts in self.tracks.values():
            for t in parts:
                if t.data is None:
                    t.load_data_from_file(self.data_dir)
                self._save_track_to_file(path, t, data_types)

                if self.mean_std:
                    for dt in running_std:
                        running_std[dt].update(t.data[:, TYPES_INDICES[dt]])

                if not save_stats and t.stats != {}:
                    save_stats = True

                t.unload_data()

        if save_stats:
            self._save_stats(path)

        if self.mean_std:
            self._save_mean_std(path, running_std)

    def _save_stats(self, path):
        stats = {}
        for parts in self.tracks.values():
            for t in parts:
                stats[t.id] = {}
                for k, v in t.stats.items():
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    stats[t.id][k] = v

        with open(f"{path}/stats.json", "w") as f:
            json.dump(stats, f)

    def _save_mean_std(self, path, running_std):
        lines = [",mean,std"] + [
            f"{k},{stat.mean()},{stat.std()}" for (k, stat) in running_std.items()
        ]

        amplitudes = [t.amplitude for parts in self.tracks.values() for t in parts]
        if amp_mean := np.mean(amplitudes) != 0:
            lines.append(f"amplitude,{amp_mean},{np.std(amplitudes)}")

        with open(f"{path}/mean_std.csv", "w") as f:
            f.write('\n'.join(lines))

    @staticmethod
    def _remove_backslash(s):
        return s.replace("/", "_")

    def _get_track_file_name(self, t: Track):
        label = self._remove_backslash(self.norad_to_label[t.norad_id])
        return f"{label}/{t.id}_{t.start_idx}_{t.end_idx}"

    def _save_track_to_file(self, path: str, t: Track, data_types=None):
        if data_types != []:
            filename = self._get_track_file_name(t)
            indices = [TYPES_INDICES[dt] for dt in data_types]
            np.savetxt(
                f"{path}/data/{filename}.csv",
                t.data[:, indices],
                delimiter=",",
                header=",".join(data_types),
                comments="",
            )

    def _save_metadata(self, path):
        metadata = ["id,label,period,amplitude,start_idx,end_idx,data"]
        for parts in self.tracks.values():
            for t in parts:
                label = self.norad_to_label[t.norad_id].replace(",", "_")
                file = f"data/{self._get_track_file_name(t)}.csv"
                columns = [t.id, label, t.period, t.amplitude, t.start_idx, t.end_idx, file]
                metadata.append(",".join(map(str, columns)))

        with open(f"{path}/{self.name}.csv", "w") as f:
            text = '\n'.join(metadata)
            f.write(text)

class RunningMeanStd:

    def __init__(self):
        self._mean = 0
        self.std2 = 0
        self.count = 0

    def update(self, data):
        x = data[data != 0]
        l = len(x)
        m = np.mean(x)
        s = np.std(x)
        self._mean = (self._mean * self.count + m * l) / (self.count + l)
        self.std2 = ((self.count - 1) * self.std2 + (l - 1) * s**2) / (self.count + l - 1) + (
            (self.count * l * (self._mean - m) ** 2) / ((self.count + l) * (self.count + l - 1))
        )
        self.count += l

    def mean(self):
        return self._mean

    def std(self):
        return np.sqrt(self.std2)