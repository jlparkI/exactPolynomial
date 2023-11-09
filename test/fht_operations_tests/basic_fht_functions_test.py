"""Tests the 'basic' hadamard transform based operations (SORF, SRHT, FHT)
for CPU and -- if GPU is available -- for GPU as well. We must test
for both single and double precision."""
import sys
import unittest
import numpy as np
from scipy.linalg import hadamard

from cpu_poly_feats import cpuSRHT as cSRHT

#try:
from cuda_poly_feats import cudaSRHT
import cupy as cp
#except:
#    pass


class TestFastHadamardTransform(unittest.TestCase):
    """Runs tests for basic functionality (i.e. FHT and SORF)
    for float and double precision for CPU and (if available) GPU."""


    def test_srht(self):
        """Tests SRHT functionality. Note that this tests SRHT
        functionality by using FHT. Therefore if the FHT did not
        pass, this one will not either."""

        outcomes = run_srht_test((150,256), 128)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_srht_test((304,512), 256)
        for outcome in outcomes:
            self.assertTrue(outcome)

        outcomes = run_srht_test((5,2048), 512)
        for outcome in outcomes:
            self.assertTrue(outcome)




def run_srht_test(dim, compression_size, random_seed = 123):
    """A helper function that runs the SRHT test for
    specified input dimensions."""
    marr_gt_double, marr_test_double, marr_gt_float, marr_test_float,\
            radem, norm_constant, sampler, scaling_factor = setup_srht_test(dim,
                                    compression_size, random_seed)

    if "cupy" in sys.modules:
        cuda_test_double = cp.asarray(marr_test_double)
        cuda_test_float = cp.asarray(marr_test_float)

    marr_gt_double = marr_gt_double * radem[None,:] * norm_constant
    cFHT2D(marr_gt_double, 2)
    marr_gt_double[:,:compression_size] = marr_gt_double[:,sampler]
    marr_gt_double[:,:compression_size] *= scaling_factor

    marr_gt_float = marr_gt_float * radem[None,:] * norm_constant
    cFHT2D(marr_gt_float, 2)
    marr_gt_float[:,:compression_size] = marr_gt_float[:,sampler]
    marr_gt_float[:,:compression_size] *= scaling_factor

    cSRHT(marr_test_double, radem, sampler, compression_size, 2)
    cSRHT(marr_test_float, radem, sampler, compression_size, 2)
    outcome_d = np.allclose(marr_gt_double, marr_test_double)
    outcome_f = np.allclose(marr_gt_float, marr_test_float)
    print("**********\nDid the C extension provide the correct result for SRHT of "
            f"a {dim} 2d array of doubles? {outcome_d}\n*******")
    print("**********\nDid the C extension provide the correct result for SRHT of "
            f"a {dim} 2d array of floats? {outcome_f}\n*******")

    if "cupy" in sys.modules:
        radem = cp.asarray(radem)
        cudaSRHT(cuda_test_float, radem, sampler, compression_size, 2)
        cudaSRHT(cuda_test_double, radem, sampler, compression_size, 2)
        cuda_test_double = cp.asnumpy(cuda_test_double)
        cuda_test_float = cp.asnumpy(cuda_test_float)
        outcome_cuda_d = np.allclose(marr_gt_double, cuda_test_double)
        outcome_cuda_f = np.allclose(marr_gt_float, cuda_test_float)
        print("**********\nDid the Cuda extension provide the correct result for SRHT of "
            f"a {dim} 2d array of doubles? {outcome_cuda_d}\n*******")
        print("**********\nDid the Cuda extension provide the correct result for SRHT of "
            f"a {dim} 2d array of floats? {outcome_cuda_f}\n*******")
        return outcome_d, outcome_f, outcome_cuda_d, outcome_cuda_f

    return outcome_d, outcome_f



def setup_srht_test(dim, compression_size, random_seed = 123):
    """A helper function that builds the matrices required for
    the SRHT test, specified using the input dimensions."""
    radem_array = np.asarray([-1.0,1.0], dtype=np.int8)
    rng = np.random.default_rng(random_seed)

    marr_gt_double = rng.uniform(low=-10.0,high=10.0, size=dim)
    marr_test_double = marr_gt_double.copy()
    marr_gt_float = marr_gt_double.copy().astype(np.float32)
    marr_test_float = marr_gt_float.copy()

    radem = rng.choice(radem_array, size=(dim[1]), replace=True)
    sampler = rng.permutation(dim[1])[:compression_size]
    norm_constant = np.log2(dim[1]) / 2
    norm_constant = 1 / (2**norm_constant)
    scaling_factor = np.sqrt(radem.shape[0] / compression_size)
    return marr_gt_double, marr_test_double, marr_gt_float, marr_test_float, \
            radem, norm_constant, sampler, scaling_factor


if __name__ == "__main__":
    unittest.main()
