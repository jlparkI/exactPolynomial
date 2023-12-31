"""Tests dataset construction and ensures that y_mean, y_std
are calculated correctly."""
import sys
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset



class CheckDatasetConstruction(unittest.TestCase):
    """Tests construction of dataset objects."""

    def test_dataset_builders(self):
        """Test the dataset builders."""
        test_online_dataset, test_offline_dataset = build_test_dataset(
            xsuffix = "testxvalues.npy", ysuffix = "testyvalues.npy")
        train_online_dataset, train_offline_dataset = build_test_dataset()

        #Access protected class members directly
        test_ymean = np.mean(test_online_dataset.ydata_)
        test_ystd = np.std(test_online_dataset.ydata_)
        train_ymean = np.mean(train_online_dataset.ydata_)
        train_ystd = np.std(train_online_dataset.ydata_)

        self.assertTrue(np.allclose(test_ymean, test_offline_dataset.get_ymean()))
        self.assertTrue(np.allclose(test_ystd, test_offline_dataset.get_ystd()))
        self.assertTrue(np.allclose(train_ymean, train_offline_dataset.get_ymean()))
        self.assertTrue(np.allclose(train_ystd, train_offline_dataset.get_ystd()))


        test_xdim = test_online_dataset.xdata_.shape
        self.assertTrue(test_xdim == test_offline_dataset.get_xdim())
        self.assertTrue(test_xdim == test_online_dataset.get_xdim())

if __name__ == "__main__":
    unittest.main()
