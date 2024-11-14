import re
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from dataset_builder.lcdataset import LCDataset

from utils import Track, load_rsos_from_csv, load_tracks_from_csv

class DatasetBuilder:
    
    def __init__(self, 
                 directory,
                 classes,
                 regexes=None,
                 preprocessing=None,
                 step=None,
                 split_ratio=None,
                 mean_std=False,
                 lazy=False):
        
        self.dir = directory
        self.preprocessing = preprocessing 
        self.step = step
        self.split_ratio = split_ratio
        self.compute_mean_std = mean_std
        self.lazy = lazy

        self.classes = {c:i for i, c in enumerate(classes)}
        regexes = regexes if regexes is not None else classes
        self.regexes = {c:re.compile(r) for c,r in zip(classes, regexes)}

        self.objects, self.tracks, self.labels = self.load_data()
        print(f"Loaded {len(self.objects)} objects and {len(self.tracks)} tracks")

    def load_data(self):
        object_list = load_rsos_from_csv(f"{self.dir}/rso.csv")
        label_list = list(map(self._get_label, object_list))

        objects = {o.norad_id: o for o, l in zip(object_list, label_list) if l is not None}
        labels  = {o.norad_id: l for o, l in zip(object_list, label_list) if l is not None}

        track_list = load_tracks_from_csv(f"{self.dir}/tracks.csv")
        tracks = {t.id: [t] for t in track_list if t.norad_id in objects}
        
        return objects, tracks, labels
    
    def _get_label(self, rso):
        for c, r in self.regexes.items():
            if r.match(rso.name) is not None:
                return c
        return None
    
    def split_train_test(self, ratio=0.8, seed=None):

        data = []
        for t_id in self.tracks:
            t = self.tracks[t_id]
            if isinstance(t, list):
                t = t[0]
            c = self.labels[t.norad_id]
            data.append([t_id, self.classes[c]])

        data = np.array(data)

        X = data[:,0]
        y = data[:, 1]

        train, test, _, _ = train_test_split(X, y, test_size=1-ratio, stratify=y, random_state=seed)

        train_tracks = {t_id: self.tracks[t_id] for t_id in train}
        test_tracks = {t_id: self.tracks[t_id] for t_id in test}

        return train_tracks, test_tracks
    
    def build_dataset(self):
        def fun(ts: List[Track]):
            t = ts[0]
            t.load_data_from_file(f"{self.dir}/data")
            parts = self.preprocessing(t, self.objects[t.norad_id])
            if self.lazy:
                for p in parts:
                    if self.lazy: p.unload_data()
            return parts
                
        if self.preprocessing is not None:
            self.tracks = {ts[0].id: res for ts in self.tracks.values() if (res := fun(ts)) != []}
            objects_to_be_removed = set(self.objects) - set([self.tracks[i][0].norad_id for i in self.tracks])
            for o_id in objects_to_be_removed:
                del self.objects[o_id]
        
        if self.split_ratio is None:
            return (LCDataset(self.tracks, self.labels,
                              self.dir, self.compute_mean_std,"data"), 
                    None)
        else:
            train_tracks, test_tracks = self.split_train_test(self.split_ratio)
            return (LCDataset(train_tracks, self.labels, 
                              self.dir, self.compute_mean_std, "train"), 
                    LCDataset(test_tracks, self.labels, 
                              self.dir, self.compute_mean_std, "test") ) 
        
