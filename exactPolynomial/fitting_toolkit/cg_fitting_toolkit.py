"""Implements functions for fitting (once hyperparameters have been
selected) using CG."""
import warnings
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as Cuda_CG
    from .cuda_cg_linear_operators import Cuda_CGLinearOperator
except:
    pass
import numpy as np
from scipy.sparse.linalg import cg as CPU_CG

from .cg_linear_operators import CPU_CGLinearOperator


def calc_zty(dataset, kernel):
    """Calculates the vector Z^T y.

    Args:
        dataset: An Dataset object that can supply
            chunked data.
        kernel: A valid kernel object that can generate
            random features.
        device (str): One of "cpu", "gpu".

    Returns:
        z_trans_y (array): A shape (num_rffs)
            array that contains Z^T y.
        y_trans_y (float): The value y^T y.
    """
    if kernel.device == "gpu":
        z_trans_y = cp.zeros((kernel.get_num_feats()))
    else:
        z_trans_y = np.zeros((kernel.get_num_feats()))

    y_trans_y = 0

    for xdata, ydata in dataset.get_chunked_data():
        zdata = kernel.transform_x(xdata)
        z_trans_y += zdata.T @ ydata
        y_trans_y += float( (ydata**2).sum() )
    return z_trans_y, y_trans_y



def cg_fit_lib_ext(kernel, dataset, cg_tol = 1e-5, max_iter = 500,
                        preconditioner = None, verbose = True):
    """Calculates the weights when fitting the model using
    preconditioned CG. Good scaling but slower for small
    numbers of random features.

    Args:
        kernel: A valid kernel object that can generate random features.
        dataset: Either OnlineDataset or OfflineDataset.
        cg_tol (float): The threshold below which cg is deemed to have
            converged. Defaults to 1e-5.
        max_iter (int): The maximum number of iterations before
            CG is deemed to have failed to converge.
        preconditioner: Either None or a valid Preconditioner (e.g.
            CudaRandomizedPreconditioner, CPURandomizedPreconditioner
            etc). If None, no preconditioning is used. Otherwise,
            the preconditioner is used for CG. The preconditioner
            can be built by calling self.build_preconditioner
            with appropriate arguments.
        verbose (bool): If True, print regular updates.

    Returns:
        weights: A cupy or numpy array of shape (M) for M
            random features.
        n_iter (int): The number of CG iterations.
        losses (list): The loss on each iteration; for diagnostic
            purposes.
    """
    if kernel.device == "gpu":
        cg_fun = Cuda_CG
        cg_operator = Cuda_CGLinearOperator(dataset, kernel,
                verbose)
    else:
        cg_fun = CPU_CG
        cg_operator = CPU_CGLinearOperator(dataset, kernel,
                verbose)

    z_trans_y, _ = calc_zty(dataset, kernel)

    weights, convergence = cg_fun(A = cg_operator, b = z_trans_y,
            M = preconditioner, tol = cg_tol, atol = 0, maxiter = max_iter)

    if convergence != 0:
        warnings.warn("Conjugate gradients failed to converge! Try refitting "
                        "the model with updated settings.")

    return weights, cg_operator.n_iter, []
