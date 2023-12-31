"""Tests LBFGS fitting."""
import sys
import unittest

import numpy as np

#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model

#A set of hyperparameters known to work well for our testing dataset
#that we can use as a default.
HPARAM = np.array([-1.378])

RANDOM_SEED = 123


class CheckLBFGSFit(unittest.TestCase):
    """Tests LBFGS fitting."""

    def test_lbfgs(self):
        """Test using LBFGS, which should easily fit in under
        150 epochs."""
        online_data, _ = build_test_dataset()
        cpu_mod, gpu_mod = get_models(online_data)

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                mode = "lbfgs", preset_hyperparams = HPARAM,
                tol = 1e-4)
        print(f"L2 regularization, niter: {niter}")
        self.assertTrue(niter < 150)

        if gpu_mod is not None:
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-4,  mode = "lbfgs", preset_hyperparams = HPARAM)
            print(f"L2 regularization, niter: {niter}")
            self.assertTrue(niter < 150)

        cpu_mod, gpu_mod = get_models(online_data, regularization = "l1")

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                mode = "lbfgs", preset_hyperparams = HPARAM,
                tol = 1e-4)
        print(f"L1_ regularization, niter: {niter}")
        self.assertTrue(niter < 250)

        if gpu_mod is not None:
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-4,  mode = "lbfgs", preset_hyperparams = HPARAM)
            print(f"L1_ regularization, niter: {niter}")
            self.assertTrue(niter < 250)


if __name__ == "__main__":
    unittest.main()
