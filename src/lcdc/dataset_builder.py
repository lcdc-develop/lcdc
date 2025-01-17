import re
from typing import List
from tqdm import tqdm
import os
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .vars import TableCols as TC, DATA_COLS
from .preprocessing import Compose


class DatasetBuilder:
    
    def __init__(self, 
                 directory,
                 classes=[],
                 regexes=[],
                 norad_ids=None):
        
        self.dir = directory
        self.norad_ids = norad_ids
        assert classes is not None or norad_ids is not None, "Either classes or norad_ids must be provided"

        self.classes = classes
        self.regexes = {}
        if regexes != []:
            self.regexes = {c:re.compile(r) for c,r in zip(classes, regexes)}
        elif classes != []:
            self.regexes = {c:re.compile(c) for c in classes}

        self.table = self.load_data()

        print(f"Loaded {len(self.table)} track")

    def load_data(self):

        parquet_files = glob.glob(f"{self.dir}/*.parquet")
        dataset = pq.ParquetDataset(parquet_files)
        table = dataset.read()


        if self.norad_ids is not None:
            mask = pc.is_in(table[TC.NORAD_ID], value_set=pa.array(self.norad_ids))
            table = table.filter(mask)

        names = list(map(lambda x: x.as_py(), table[TC.NAME]))
        labels = list(map(self._get_label, names))
        table = table.append_column(TC.LABEL, pa.array(labels))

        if self.classes != []:
            table = table.filter(pc.field(TC.LABEL) != 'Unknown')

        ranges = [(0,len(x)-1) for x in table[TC.TIME]]
        table = table.append_column(TC.RANGE, pa.array(ranges))

        return table
    
    def _get_label(self, name):
        for c, r in self.regexes.items():
            if r.match(name) is not None:
                return c
        return 'Unknown'
    
    def split_train_test(self, ratio=0.8, seed=None):

        data = []
        table = self.table
        data = ( table.group_by(TC.ID, use_threads=False)
                      .aggregate([(TC.LABEL, "first")])
                      .to_pandas()
                      .to_numpy() )
        
        X = data[:,0]
        y = data[:, 1]

        train_id, test_id, _, _ = train_test_split(X, y, test_size=1-ratio, stratify=y, random_state=seed)

        return train_id, test_id
    
    def preprocess(self, ops=[]):
        if ops == []:
            return 
        
        preprocessor = Compose(*ops)

        table = self.table
        new_table = None
        for i in range(len(table)):
            t = table.slice(i,1).to_pylist()[0]
            for c in DATA_COLS:
                if c in t:
                    t[c] = np.array(t[c])

            records = preprocessor(t)
            if records != []:
                if new_table is None:
                    new_table = pa.Table.from_pylist(records)
                else:
                    new_table = pa.concat_tables([new_table, pa.Table.from_pylist(records)])

        self.table = new_table 
    
    def build_dataset(self, split_ratio=None):
        
        datasets = []
        if split_ratio is None:
            datasets = Dataset.from_dict(self.table.to_pydict())
        else:
            train_id, test_id = self.split_train_test(split_ratio)
            train_mask = pc.is_in(self.table[TC.ID], pa.array(train_id))
            train_table = self.table.filter(train_mask)
            test_mask = pc.is_in(self.table[TC.ID], pa.array(test_id))
            test_table = self.table.filter(test_mask)
            datasets = [
                Dataset.from_dict(train_table.to_pydict()),
                Dataset.from_dict(test_table.to_pydict())
            ]

        return datasets
    
    def to_file(self, path):
        pq.write_table(self.table, f"{path}.parquet")

    @staticmethod
    def from_file(path):
        table = pq.read_table(path)
        instance = DatasetBuilder.__new__(DatasetBuilder)
        instance.table = table
        instance.dir = None
        instance.classes = None
        instance.regexes = None
        instance.norad_ids = None
        return instance

