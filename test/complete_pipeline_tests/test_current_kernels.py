"""Tests using both exact fitting and preconditioned CG with
minimal bayes tuning to ensure that we can achieve expected
performance for all currently implemented kernels. This is an
'all-in-one' workflow test, if it fails, run fitting tests,
tuning tests, preconditioner tests and fht operations tests
as appopriate to determine which component is failing."""
import unittest

from current_kernel_list import IMPLEMENTED_KERNELS
from fitting_utils import test_fit

RANDOM_SEED = 123


class CheckPipeline(unittest.TestCase):
    """An all in one pipeline test."""



    def test_fit_cpu(self):
        """Test on cpu."""
        print("Now running CPU tests.")
        for regularization in ["l1", "l2"]:
            for interaction in [True, False]:
                for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
                    score = test_fit(device="cpu", regularization = regularization,
                            interactions_only = interaction)
                    self.assertTrue(score > exp_score)

    def test_fit_gpu(self):
        """Test on gpu."""
        print("Now running GPU tests.")
        for regularization in ["l1", "l2"]:
            for interaction in [True, False]:
                for kernel_name, (is_conv, exp_score) in IMPLEMENTED_KERNELS.items():
                    score = test_fit(device="gpu", regularization = regularization,
                            interactions_only = interaction)
                    self.assertTrue(score > exp_score)


if __name__ == "__main__":
    unittest.main()
