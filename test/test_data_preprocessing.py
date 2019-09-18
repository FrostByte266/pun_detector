import unittest
from unittest import mock
import requests
import sys 
import csv
import numpy as np
import os

sys.path.append('/src/')

import data_preprocessing

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.puns = data_preprocessing.fetch_puns_list(1)

    # Tests related to fetching data

    @mock.patch('requests.get')
    def test_data_fetching_not_found(self, mocked_get):
        type(mocked_get.return_value).status_code = mock.PropertyMock(return_value=404)
        with self.assertRaises(RuntimeError):
            data_preprocessing.fetch_puns_list(1)

    def test_data_fetching_ok(self):
        self.assertNotEqual(len(self.puns), 0)

    def test_no_none_in_fetch(self):
        self.assertNotIn(None, self.puns)

    # Tests related to loading datasets

    def test_csv_has_two_columns(self):
        data = data_preprocessing.read_dataset()
        self.assertEqual(data.shape[1], 2)

    def test_loading_file_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            data_preprocessing.read_dataset(dataset='/some/file/that/doesnt/exist', on_error='raise')

    def test_file_loads(self):
        data = data_preprocessing.read_dataset()
        self.assertEqual(type(data), type(np.array([])))

    def test_file_is_created(self):
        tests = []
        path = '/test/training.csv'
        data = data_preprocessing.read_dataset(path, num_pages=1)
        tests.append(True if os.path.exists(path) else False)
        tests.append(True if type(data) is type(np.array([])) else False)
        os.remove(path)
        self.assertTrue(all(tests))
