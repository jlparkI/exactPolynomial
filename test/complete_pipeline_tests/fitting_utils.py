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




def test_fit(device = "gpu", regularization = "l1"):
    """Test on a specified device using preconditioned CG and exact."""
    _, train_dataset = build_test_dataset()
    cpu_mod, gpu_mod = get_models(train_dataset, regularization = regularization)
    if device == "gpu":
        if gpu_mod is None:
            #If GPU not available, return immediately.
            return None, None
        else:
            model = gpu_mod
    else:
        model = cpu_mod

    model.verbose = True

    test_dataset, _ = build_test_dataset(xsuffix = "testxvalues.npy",
            ysuffix = "testyvalues.npy")

    hparams = np.array([-0.687])

    if regularization == "l2":
        preconditioner, ratio = model.build_preconditioner(train_dataset, max_rank = 256,
            preset_hyperparams = hparams)
    else:
        preconditioner = None

    model.fit(train_dataset, preset_hyperparams = hparams, max_iter = 1000,
            mode = "ista", tol=1e-3, preconditioner = preconditioner)
    score = evaluate_model(model, train_dataset, test_dataset)

    print(f"Test set score, {score}")

    return score
