"""Contains the tools needed to get weights for the model
using the L-BFGS optimization algorithm."""
import numpy as np
try:
    import cupy as cp
except:
    pass
from scipy.optimize import minimize



class lBFGSModelFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-BFGS
    algorithm.

    Attributes:
        dataset: An OnlineDataset or OfflineDatset containing all the
            training data.
        kernel: A kernel object that can generate random features for
            the Dataset.
        lambda_ (float): The noise hyperparameter shared across all kernels.
        verbose (bool): If True, print regular updates.
        device (str): One of 'cpu', 'gpu'. Indicates where calculations
            will be performed.
        zero_arr: A convenience reference to either cp.zeros or np.zeros.
        dtype: A convenience reference to either cp.float64 or np.float64,
            depending on device.
        signval: A convenience reference to either cp.sign or np.sign.
        niter (int): The number of function evaluations performed.
        eps (float): A small value added to the square of each weight
            if using l1 regularization; this approximation to the l1
            penalty ensures it is differentiable. The default is usually
            fine.
    """

    def __init__(self, dataset, kernel, device, verbose, eps = 1e-8):
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
            eps (float): A small value added to the square of each weight
                if using l1 regularization; this approximation to the l1
                penalty ensures it is differentiable. The default is usually
                fine.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
        self.n_iter = 0
        self.eps = eps


    def fit_model(self, max_iter = 500, tol = 3e-09, regularization = "l2", hard_thresh = 1e-6):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence.
            regularization (str): One of 'l1', 'l2'.
            hard_thresh (float): The value below which weights are set to zero when using l1
                regularization at the end of fitting.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_feats().
        """
        init_weights = np.zeros((self.kernel.get_num_feats()))
        z_trans_y, y_trans_y = calc_zty(self.dataset, self.kernel)

        if regularization == 'l2':
            res = minimize(self.cost_fun, options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B",
                    x0 = init_weights, args = (z_trans_y, y_trans_y),
                    jac = True, bounds = None)
            wvec = res.x
        else:
            res = minimize(self.l1_cost_fun, options={"maxiter":max_iter, "ftol":tol},
                    method = "L-BFGS-B", args = (z_trans_y,),
                    x0 = init_weights, jac = True, bounds = None)
            wvec = res.x
            wvec[np.abs(wvec) < hard_thresh] = 0.

        if self.device == "gpu":
            wvec = cp.asarray(wvec)
        return wvec, self.n_iter, []


    def cost_fun(self, weights, z_trans_y, y_trans_y):
        """The cost function for finding the weights using
        L-BFGS. Returns both the current loss and the gradient.

        Args:
            weights (np.ndarray): The current set of weights.
            z_trans_y: A cupy or numpy array (depending on device)
                containing Z.T @ y, where Z is the random features
                generated for all of the training datapoints.
            y_trans_y (float): y^T y.

        Returns:
            loss (float): The current loss.
            grad (np.ndarray): The gradient for the current set of weights.
        """
        wvec = weights
        if self.device == "gpu":
            wvec = cp.asarray(wvec).astype(self.dtype)
        xprod = self.lambda_**2 * wvec

        for xdata in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata)
            xprod += (xtrans.T @ (xtrans @ wvec))

        grad = 2 * xprod - 2 * z_trans_y
        loss = float(y_trans_y + (wvec.T @ xprod) - 2 * wvec.T @ z_trans_y)


        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete, loss {loss}", flush=True)
        self.n_iter += 1
        return loss, grad



    def l1_cost_fun(self, weights, z_trans_y):
        """The cost function for finding the weights using
        L-BFGS when using l1 regularization. The l1 penalty
        is approximated to make it differentiable.
        Returns both the current loss and the gradient.

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
            loss = (cp.sqrt(wvec**2 + self.eps) * self.lambda_**2).sum()
            grad = -2 * z_trans_y + self.lambda_**2 * wvec / cp.sqrt(wvec**2 + self.eps)
        else:
            loss = (np.sqrt(wvec**2 + self.eps) * self.lambda_**2).sum()
            grad = -2 * z_trans_y + self.lambda_**2 * wvec / np.sqrt(wvec**2 + self.eps)

        for xdata, ydata in self.dataset.get_chunked_data():
            xtrans = self.kernel.transform_x(xdata)
            loss += ((ydata - xtrans @ wvec)**2).sum()
            grad += 2 * (xtrans.T @ (xtrans @ wvec))

        loss = float(loss)

        if self.device == "gpu":
            grad = cp.asnumpy(grad).astype(np.float64)
        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter} complete, loss {loss}", flush=True)
        self.n_iter += 1
        return loss, grad



def calc_zty(dataset, kernel):
    """Calculates the vector Z^T y.

    Args:
        dataset: An Dataset object that can supply
            chunked data.
        kernel: A valid kernel object that can generate
            random features.
        device (str): One of "cpu", "gpu".

    Returns:
        z_trans_y (array): A shape (num_feats)
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
