"""Tests CG fitting."""
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


class CheckCGFit(unittest.TestCase):
    """Tests CG fitting."""

    def test_cg(self):
        """Test using preconditioned CG, which should easily fit in under
        150 epochs."""
        online_data, _ = build_test_dataset()
        cpu_mod, gpu_mod = get_models(online_data)

        preconditioner, _ = cpu_mod.build_preconditioner(online_data,
                max_rank = 256, preset_hyperparams = HPARAM)

        niter, _ = cpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                mode = "cg", preset_hyperparams = HPARAM,
                tol = 1e-5, preconditioner = preconditioner)
        print(f"niter: {niter}")
        self.assertTrue(niter < 150)

        if gpu_mod is not None:
            preconditioner, _ = gpu_mod.build_preconditioner(online_data,
                max_rank = 256, preset_hyperparams = HPARAM)
            niter, _ = gpu_mod.fit(online_data,
                max_iter = 500, run_diagnostics = True,
                tol = 1e-5,  mode = "cg", preconditioner = preconditioner,
                preset_hyperparams = HPARAM)
            print(f"niter: {niter}")
            self.assertTrue(niter < 150)


if __name__ == "__main__":
    unittest.main()
