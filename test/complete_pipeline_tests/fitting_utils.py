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




def test_fit(device = "gpu", regularization = "l2", interactions_only = False):
    """Test on a specified device using preconditioned CG and exact."""
    train_dataset, _ = build_test_dataset()
    cpu_mod, gpu_mod = get_models(train_dataset, regularization = regularization,
            interactions_only = interactions_only)
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

    hparams = np.array([0.])

    if regularization == "l2":
        preconditioner, _ = model.build_preconditioner(train_dataset, max_rank = 256,
            preset_hyperparams = hparams)
        model.fit(train_dataset, preset_hyperparams = hparams, max_iter = 2000,
            mode = "cg", tol=1e-6, preconditioner = preconditioner)
    else:
        model.fit(train_dataset, preset_hyperparams = hparams, max_iter = 2000,
            mode = "lbfgs", tol=1e-6, preconditioner = None)
    score = evaluate_model(model, train_dataset, test_dataset)

    print(f"{regularization}, interactions {interactions_only}, Test set score, {score}")

    return score
