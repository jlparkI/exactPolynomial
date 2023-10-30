"""Tests grid_bfgs fitting to ensure we achieve performance
>= what has been seen in the past for a similar # of RFFs and
kernel. Tests either CG or exact fitting."""
import sys
import numpy as np
#TODO: Get rid of this path alteration
sys.path.append("..")
from utils.build_test_dataset import build_test_dataset
from utils.model_constructor import get_models
from utils.evaluate_model import evaluate_model


RANDOM_STATE = 123


def test_fit(device = "gpu"):
    """Test on a specified device using preconditioned CG and exact."""
    _, train_dataset = build_test_dataset()
    cpu_mod, gpu_mod = get_models(train_dataset)
    if device == "gpu":
        if gpu_mod is None:
            #If GPU not available, return immediately.
            return None, None
        else:
            model = gpu_mod
    else:
        model = cpu_mod

    model.verbose = False

    test_dataset, _ = build_test_dataset(xsuffix = "testxvalues.npy",
            ysuffix = "testyvalues.npy")

    hparams = np.array([-1.375])


    model.fit(train_dataset, preset_hyperparams = hparams, max_iter = 500,
            mode = "lbfgs")
    score = evaluate_model(model, train_dataset, test_dataset)

    print(f"LBFGS score, {score}")

    return score
