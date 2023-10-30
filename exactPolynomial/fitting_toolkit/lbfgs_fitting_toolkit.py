"""Contains the tools needed to get weights for the model
using the L-BFGS optimization algorithm."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from scipy.optimize import minimize

from ..scoring_toolkit.exact_nmll_calcs import calc_zty



class lBFGSModelFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-BFGS
    algorithm.

    Attributes:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        regularization (str): One of 'l1', 'l2'. Determines the type
            of regularization that is applied.
        kernel: A kernel object that can generate random features for
            the Dataset.
        lambda_ (float): The noise hyperparameter shared across all kernels.
        verbose (bool): If True, print regular updates.
        device (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        zero_arr: A convenience reference to either cp.zeros or np.zeros.
        dtype: A convenience reference to either cp.float64 or np.float64,
            depending on device.
        absval: A convenience reference to either cp.abs or np.abs.
        niter (int): The number of function evaluations performed.
    """

    def __init__(self, dataset, regularization, kernel, device, verbose):
        """Class constructor.

        Args:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        regularization (str): One of 'l1', 'l2'. Determines the type of
            regularization that is applied.
        kernel: A kernel object that can generate random features for
            the Dataset.
        device (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        verbose (bool): If True, print regular updates.
        """
        if regularization not in ['l1', 'l2']:
            raise ValueError("Unrecognized regularization option supplied.")
        self.dataset = dataset
        self.regularization = regularization
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
            self.absval = cp.abs
        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
            self.absval = np.abs
        self.n_iter = 0

    def fit_model_lbfgs(self, max_iter = 500, tol = 3e-09):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence. User not currently
                allowed to specify since setting a larger / smaller tol
                can result in very poor results with L-BFGS (either very
                large number of iterations or poor performance).

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_rffs().
        """
        z_trans_y, _ = calc_zty(self.dataset, self.kernel)
        init_weights = np.zeros((self.kernel.get_num_rffs()))
        res = minimize(self.cost_fun, options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B",
                    x0 = init_weights, args = (z_trans_y,),
                    jac = True, bounds = None)

        wvec = res.x
        if self.device == "gpu":
            wvec = cp.asarray(wvec)
        return wvec, self.n_iter, []


    def cost_fun(self, weights, z_trans_y):
        """The cost function for finding the weights using
        L-BFGS. Returns both the current loss and the gradient.

        Args:
            weights (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
        if self.regularization == "l1":
            xprod = self.lambda_**2 * wvec
        else:
            xprod = self.lambda_ * self.absval(wvec)

        for xdata in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata)
            xprod += (xtrans.T @ (xtrans @ wvec))


        grad = xprod - z_trans_y
        loss = 0.5 * (wvec.T @ xprod) - z_trans_y.T @ wvec


        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete")
        self.n_iter += 1
        return float(loss), grad