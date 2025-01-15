import re
from typing import List
from tqdm import tqdm
import os
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .lcdataset import LCDataset
from .vars import TableCols as TC


class DatasetBuilder:
    
    def __init__(self, 
                 directory,
                 classes=None,
                 regexes=None,
                 norad_ids=None,
                 preprocessing=None,
                 statistics=[],
                 split_ratio=None,
                 mean_std=False,
                 lazy=False):
        
        self.dir = directory
        self.preprocessing = preprocessing
        self.statistics = statistics
        self.split_ratio = split_ratio
        self.compute_mean_std = mean_std
        self.lazy = lazy

        self.norad_ids = norad_ids
        assert classes is not None or norad_ids is not None, "Either classes or norad_ids must be provided"

        if norad_ids is not None:
            classes = "Unknown"
            regexes = [".*"]

        self.classes = {c:i for i, c in enumerate(classes)}


        regexes = regexes if regexes is not None else classes
        self.regexes = {c:re.compile(r) for c,r in zip(classes, regexes)}
        self.table = self.load_data()



        print(f"Loaded {len(self.objects)} objects and {len(self.tracks)} tracks")

    def load_data(self):

        parquet_files = glob.glob(f"{self.dir}/*.parquet")
        dataset = pq.ParquetDataset(parquet_files)
        table = dataset.read()


        if self.norad_ids is not None:
            mask = pc.is_in(table[TC.NORAD_ID], value_set=self.norad_ids)
            table = table.filter(mask)

        names = list(map(lambda x: x.as_py(), table[TC.NAME]))
        labels = list(map(self._get_label, names))
        table = table.append_column(TC.LABEL, pa.array(labels))

        if self.classes is not None:
            table = table.filter(pc.field(TC.LABEL) != 'Unknown')

        splits = [(0,len(x)) for x in table[TC.DATA]]
        table = table.append_column(TC.RANGE, pa.array(splits))

        return table
    
    def _get_label(self, rso):
        for c, r in self.regexes.items():
            if r.match(rso.name) is not None:
                return c
        return 'Unknown'
    
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
        test_tracks  = {t_id: self.tracks[t_id] for t_id in test}

        return train_tracks, test_tracks
    
    def build_dataset(self) -> List[LCDataset]:
        
        table = self.table
        if self.preprocessing is not None:

            new_table = None
            for i in range(len(table)):
                splits = table[TC.SPLITS][i].as_py()
                t = table.splice(i,1).to_pydict()
                data = t[TC.DATA].values.to_numpy(zero_copy_only=False)
                for a,b in splits:
                    t2 = t.copy()
                    t2[TC.DATA] = np.array(data[a:b])
                    records = self.preprocessing(t2)
                    if records != []:
                        if new_table is None:
                            new_table = pa.table(records)
                        else:
                            new_table = pa.concat_tables([new_table, pa.table(records)])

            self.table = new_table 
        
        # TODO: Splitting
        datasets = []
        if self.split_ratio is None:
            datasets = [None]
        else:
            train_tracks, test_tracks = self.split_train_test(self.split_ratio)
            datasets = [None, None]

        return datasets
    
    def to_file(self, path, data_types=[]):
        datasets = self.build_dataset()

        for d in datasets:
            d.to_file(path, data_types)
    
    def to_dict(self, data_types=[]):
        datasets = self.build_dataset()
        if len(datasets) == 1:
            return datasets[0].to_dict(data_types)
        return [d.to_dict(data_types) for d in datasets]
