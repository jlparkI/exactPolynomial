"""Builds xGPRegression models with generic kernel parameters for
use in other tests."""
import sys
import copy

from exactPolynomial import ExactQuadratic as ExactQuad

RANDOM_STATE = 123


def get_models(dataset):
    """Generates a CPU model and a GPU model with generic
    kernel settings."""
    cpu_mod = ExactQuad(device = "cpu")

    if "cupy" not in sys.modules:
        print("Cupy not installed -- skipping the CUDA test.")
        gpu_mod = None
    else:
        gpu_mod = copy.deepcopy(cpu_mod)
        gpu_mod.device = "gpu"
        gpu_mod.initialize(dataset, RANDOM_STATE)

    cpu_mod.initialize(dataset, RANDOM_STATE)
    return cpu_mod, gpu_mod
