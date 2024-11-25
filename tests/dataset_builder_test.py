import os
import shutil
import tempfile
import unittest
from collections import Counter

import pandas as pd
from lcdc import DatasetBuilder, Track
from lcdc.vars import Variability


class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.rso_csv = os.path.join(self.test_dir, 'rso.csv')
        self.tracks_csv = os.path.join(self.test_dir, 'tracks.csv')

        # Create dummy RSO data
        rso_data = pd.DataFrame({
            'mmt_id': [1, 2, 3],
            'norad_id': [1, 2, 3],
            'name': ["A", "A", "B"],
            'country': ['USA-1', 'Russia-1', 'China-1'],
            'variability': [
                Variability.PERIODIC,
                Variability.APERIODIC,
                Variability.NONVARIABLE,
        ],})
        rso_data.to_csv(self.rso_csv, index=False)

        # Create dummy Track data
        track_data = pd.DataFrame({
            'id': [1, 2, 3],
            'norad_id': [1, 2, 3],
            'timestamp': [1, 1, 1],
            'mjd': [0, 0, 0],
            'period': [10, 20, 30],
        })

        track_data.to_csv(self.tracks_csv, index=False)

        self.classes = ["A", "B"]
        self.regexes = ["A", "B"]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_data(self):
        builder = DatasetBuilder(self.test_dir, self.classes, self.regexes)
        print(builder.objects)
        print(builder.tracks)
        print(builder.labels)
        self.assertEqual(len(builder.objects), 3)
        self.assertEqual(len(builder.tracks), 3)
        self.assertEqual(len(builder.labels), 3)

    def test_split_train_test(self):

        builder = DatasetBuilder(self.test_dir, self.classes, self.regexes)

        builder.tracks = {}
        for i in range(50):
            builder.tracks[i] = [Track(i, 1, 0, 0, 0)]
            builder.tracks[i + 50] = [Track(i, 2, 0, 0, 0)]

        builder.labels = {1: "A", 2: "B"}

        train_tracks, test_tracks = builder.split_train_test(ratio=0.5, seed=42)
        self.assertEqual(len(train_tracks), len(test_tracks))

        train_tracks, test_tracks = builder.split_train_test(ratio=0.8, seed=42)
        self.assertEqual(len(train_tracks), 80)
        self.assertEqual(len(test_tracks), 20)

        # Track is split into two parts
        builder.tracks[i] = [Track(i, 1, 0, 0, 0), Track(i, 1, 0, 0, 0)]
        builder.tracks[50] = [Track(i, 2, 0, 0, 0), Track(i, 2, 0, 0, 0)]
        train_tracks, test_tracks = builder.split_train_test(ratio=0.5, seed=42)
        self.assertEqual(len(train_tracks), len(test_tracks))

        # test stratification
        builder.tracks = {}
        for i in range(80):
            builder.tracks[i] = [Track(i, 1, 0, 0, 0)]

        for i in range(20):
            builder.tracks[i + 80] = [Track(i + 80, 2, 0, 0, 0)]

        train_tracks, test_tracks = builder.split_train_test(ratio=0.5, seed=42)

        c = Counter([t[0].norad_id for t in train_tracks.values()])
        self.assertEqual(c[1], 40)
        self.assertEqual(c[2], 10)

        c = Counter([t[0].norad_id for t in test_tracks.values()])
        self.assertEqual(c[1], 40)
        self.assertEqual(c[2], 10)

    def test_build_dataset(self):

        builder = DatasetBuilder(self.test_dir, self.classes, self.regexes, preprocessing=None)
        train_dataset, test_dataset = builder.build_dataset()

        self.assertIsNotNone(train_dataset)
        self.assertEqual(len(train_dataset.tracks), 3)
        self.assertIsNone(test_dataset)

        builder.tracks = {}
        for i in range(50):
            builder.tracks[i] = [Track(i, 1, 0, 0, 0)]
            builder.tracks[i + 50] = [Track(i, 2, 0, 0, 0)]

        builder.labels = {1: "A", 2: "B"}

        builder.split_ratio = 0.5
        train_dataset, test_dataset = builder.build_dataset()

        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(test_dataset)
        self.assertEqual(len(train_dataset.tracks), 50)
        self.assertEqual(len(test_dataset.tracks), 50)

    def test_build_dataset_with_split(self):
        builder = DatasetBuilder(self.test_dir, self.classes, self.regexes, preprocessing=None)
        builder.tracks = {}
        for i in range(50):
            builder.tracks[i] = [Track(i, 1, 0, 0, 0)]
            builder.tracks[i + 50] = [Track(i, 2, 0, 0, 0)]

        builder.labels = {1: "A", 2: "B"}

        builder.split_ratio = 0.5
        train_dataset, test_dataset = builder.build_dataset()

        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(test_dataset)
        self.assertEqual(len(train_dataset.tracks), 50)
        self.assertEqual(len(test_dataset.tracks), 50)

        # print(train_dataset.to_dict(data_types=["time", "mag"]))

if __name__ == '__main__':
    unittest.main()