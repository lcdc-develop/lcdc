import os
import shutil
import tempfile
import unittest

import numpy as np
from dataset_builder.lcdataset import LCDataset
from utils import ALL_TYPES, DataType, Track


def mock_load_data_from_file(t):
    t.data = np.zeros((100, 5))


class TestDatasetBuilder(unittest.TestCase):

    def get_dataset(self):
        tracks = {}
        norad_to_label = {1: "A", 2: "B"}

        for i in range(50):
            t = Track(i, 1, 0, 0, 0, data=np.random.randn(100, 5) + 10)
            tracks[i] = [t]
            t = Track(i + 50, 2, 0, 0, 0, data=np.random.randn(100, 5) + 10)
            tracks[i + 50] = [t]

        self.data_dir = tempfile.mkdtemp()

        return LCDataset(tracks, norad_to_label, self.data_dir, False, "test")

    def setUp(self):

        self.data_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_to_dict(self):
        d = self.get_dataset()

        r = d.to_dict(data_types=ALL_TYPES)
        for dt in ALL_TYPES:
            self.assertTrue(dt in r["data"])
            self.assertEqual(len(r["data"][dt]), 100)

        r = d.to_dict(data_types=[DataType.TIME, DataType.MAG])
        self.assertTrue(DataType.TIME in r["data"])
        self.assertTrue(DataType.MAG in r["data"])
        self.assertFalse(DataType.PHASE in r["data"])
        print(r.keys(), r["data"].keys())

    def test_to_dict_std_mean(self):
        d = self.get_dataset()
        d.mean_std = True

        r = d.to_dict(data_types=[DataType.MAG])
        self.assertTrue(DataType.MAG in r["data"])
        self.assertTrue("mean" in r)
        self.assertTrue("std" in r)
        self.assertAlmostEqual(r["mean"][DataType.MAG], 10, 1)
        self.assertAlmostEqual(r["std"][DataType.MAG], 1, 1)

    def test_to_file(self):
        d = self.get_dataset()
        d.mean_std = True

        d.to_file(self.data_dir, data_types=ALL_TYPES)

        files = set(os.listdir(self.data_dir))
        self.assertEqual(files, {"data", "mean_std.csv", "test.csv"})
        files = set(os.listdir(f"{self.data_dir}/data"))
        self.assertEqual(files, {"A", "B"})

        for i, l in enumerate("AB"):
            files = set(os.listdir(f"{self.data_dir}/data/{l}"))
            should_have = {f"{j+i*50}_0_-1.csv" for j in range(50)}
            self.assertEqual(files, should_have)

        with open(f"{self.data_dir}/test.csv", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 101)
            self.assertEqual(lines[0], "id,label,period,amplitude,start_idx,end_idx,data\n")

        with open(f"{self.data_dir}/data/A/1_0_-1.csv", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 101)
            self.assertEqual(lines[0], "time,mag,phase,distance,filter\n")

        with open(f"{self.data_dir}/mean_std.csv", "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], ",mean,std\n")
            for i in range(len(ALL_TYPES[1:-1])):
                l, m, s = lines[i + 1].split(',')
                self.assertAlmostEqual(float(m), 10, 1)
                self.assertAlmostEqual(float(s), 1, 1)
                self.assertEqual(l, ALL_TYPES[i + 1])

    def test_from_file(self):
        d = self.get_dataset()
        d.mean_std = True

        dict2 = d.to_dict(ALL_TYPES)

        d.to_file(self.data_dir, data_types=ALL_TYPES)
        dict1 = LCDataset.dict_from_file(self.data_dir)

        self.assertListEqual(list(dict1["data"].keys()), list(dict2["data"].keys()))

        for k in dict1["data"]:
            if k in ALL_TYPES:
                for i in range(len(dict1["data"][k])):
                    self.assertTrue(np.all(dict1["data"][k][i] == dict2["data"][k][i]))
            else:
                self.assertTrue(dict1["data"][k] == dict2["data"][k])

if __name__ == '__main__':
    unittest.main()