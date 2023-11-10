"""Contains the tools needed to get weights for the model
using the L-SR1 optimization algorithm with an optional
but strongly recommended preconditioner as H0."""
import numpy as np
try:
    import cupy as cp
except:
    pass

from .l1_reg_lsr1_toolkit import calc_zty



class L2_lSR1:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using the L-SR1
    algorithm. It is highly preferable to supply a preconditioner
    which acts as an H0 approximation, since otherwise this
    algorithm may take a long time to converge.

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
        n_iter (int): The number of function evaluations performed.
        n_updates (int): The number of Hessian updates to date. Not necessarily
            the same as n_iter, since Hessian updates can be skipped.
        history_size (int): The number of previous gradient updates to store.
        losses (list): A list of loss values. Useful for comparing rate of
            convergence with other options.
        preconditioner: Either None or a valid preconditioner object.
        stored_mvecs (ndarray): A cupy or numpy array containing the
            sk - Hk yk terms; shape (num_rffs, history_size).
        stored_nconstants (ndarray): The denominator of the Hessian
            update; shape (history_size).
    """

    def __init__(self, dataset, kernel, device, verbose, preconditioner = None,
            history_size = 200):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDatset containing all the
                training data.
            kernel: A kernel object that can generate random features for
                the Dataset.
            device (str): One of 'cpu', 'gpu'. Indicates where calculations
                will be performed.
            verbose (bool): If True, print regular updates.
            preconditioner: Either None or a valid preconditioner object.
            history_size (int): The number of recent gradient updates
                to store.
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
        self.n_updates = 0
        self.history_size = history_size

        self.losses = []
        self.preconditioner = preconditioner
        self.stored_mvecs = self.zero_arr((self.kernel.get_num_feats(),
            self.history_size))
        self.stored_nconstants = self.zero_arr((self.history_size))
        self.stored_nconstants[:] = 1

        self.stored_bvecs = self.stored_mvecs.copy()
        self.stored_bconstants = self.stored_nconstants.copy()


    def fit_model(self, max_iter = 500, tol = 1e-6):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations for L_BFGS.
            tol (float): The threshold for convergence.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_feats().
        """
        z_trans_y, _ = calc_zty(self.dataset, self.kernel)

        wvec = self.zero_arr((self.kernel.get_num_feats()))
        grad = -z_trans_y.copy()
        init_norms = max(np.sqrt(float((z_trans_y**2).sum()), 1e-12))


        self.n_iter, self.n_updates = 0, 0
        grad_norms = [1.]

        while self.n_iter < max_iter:
            grad, step_size = self.update_params(grad, wvec, z_trans_y)
            grad_norms.append(np.sqrt(float(grad.T @ grad)) / init_norms)
            if grad_norms[-1] < tol:
                break
            if self.verbose:
                print(f"Squared grad norm, normalized: {grad_norms[-1]}, step_size {step_size}", flush=True)
            self.n_iter += 1

        if self.device == "gpu":
            wvec = cp.asarray(wvec)
        return wvec, self.n_iter, grad_norms




    def update_params(self, grad, wvec, z_trans_y):
        """Updates the weight vector and the approximate hessian maintained
        by the algorithm.

        Args:
            grad (ndarray): The previous gradient of the weights. Shape (num_rffs).
            wvec (ndarray): The current weight values.
            z_trans_y (ndarray): The right hand side b in the equation Ax=b.

        Returns:
            wvec (ndarray): The updated wvec.
            last_wvec (ndarray): Current wvec (which is now the last wvec).
        """
        new_wvec = self.zero_arr((wvec.shape[0], 2))
        s_k = -self.inv_hess_vector_prod(grad)
        new_wvec[:,1] = s_k
        new_wvec[:,0] = wvec

        grad_update = self.cost_fun_regression(new_wvec)
        self.update_hess(grad_update.sum(axis=1) - z_trans_y - grad, s_k)

        step_size = float(z_trans_y.T @ s_k - s_k.T @ grad_update[:,0])
        denom = float(s_k.T @ grad_update[:,1])
        if np.abs(denom) < 1e-14:
            step_size = 1e-10
        else:
            step_size /= denom
        new_grad = (grad_update[:,0] + grad_update[:,1] * step_size) - \
                z_trans_y

        s_k *= step_size

        wvec += s_k
        return new_grad, step_size



    def update_hess(self, y_k, s_k):
        """Updates the hessian approximation.

        Args:
            y_k (ndarray): The shift in gradient on the current iteration.
            s_k (ndarray): The shift in the weight vector.
        """
        if self.n_updates >= self.history_size:
            return
        y_kh = self.inv_hess_vector_prod(y_k)
        s_kb = self.hess_vector_prod(s_k)

        h_update = s_k - y_kh
        b_update = y_k - s_kb
        criterion_lhs = np.abs(float(s_k.T @ (y_k - s_kb)))
        criterion_rhs = 1e-8 * float(s_k.T @ s_k) * float(b_update.T @ b_update)

        if criterion_lhs >= criterion_rhs:
            self.stored_mvecs[:,self.n_updates] = h_update
            self.stored_nconstants[self.n_updates] = h_update.T @ y_k
            self.stored_bvecs[:,self.n_updates] = b_update
            self.stored_bconstants[self.n_updates] = b_update.T @ s_k
            self.n_updates += 1


    def inv_hess_vector_prod(self, ivec):
        """Takes the product of the inverse of the approximate
        Hessian with an input vector.

        Args:
            ivec (ndarray): A cupy or numpy array of shape (num_rffs) with which
                to take the product.

        Returns:
            ovec (ndarray): Hk @ ivec.
        """
        if self.n_updates == 0:
            if self.preconditioner is None:
                return ivec
            return self.preconditioner.batch_matvec(ivec[:,None])[:,0]

        ovec = (self.stored_mvecs[:,:self.n_updates] * ivec[:,None]).sum(axis=0) / \
                self.stored_nconstants[:self.n_updates]
        ovec = (ovec[None,:] * self.stored_mvecs[:,:self.n_updates]).sum(axis=1)
        if self.preconditioner is not None:
            ovec += self.preconditioner.batch_matvec(ivec[:,None])[:,0]
        else:
            ovec += ivec
        return ovec


    def hess_vector_prod(self, ivec):
        """Takes the product of the approximate Hessian with
        an input vector.

        Args:
            ivec (ndarray): A cupy or numpy array of shape (num_rffs) with which
                to take the product.

        Returns:
            ovec (ndarray): Hk @ ivec.
        """
        if self.n_updates == 0:
            if self.preconditioner is None:
                return ivec
            return self.preconditioner.rev_batch_matvec(ivec[:,None])[:,0]

        ovec = (self.stored_bvecs[:,:self.n_updates] * ivec[:,None]).sum(axis=0) / \
                self.stored_bconstants[:self.n_updates]
        ovec = (ovec[None,:] * self.stored_bvecs[:,:self.n_updates]).sum(axis=1)
        if self.preconditioner is not None:
            ovec += self.preconditioner.rev_batch_matvec(ivec[:,None])[:,0]
        else:
            ovec += ivec
        return ovec


    def cost_fun_regression(self, wvec):
        """The cost function for finding the weights for
        regression. Returns an adjustable calculation for the
        gradient so that caller can determine step size based
        on the results.

        Args:
            wvec (np.ndarray): A (num_rffs, 2) shape array. The first
                column is the current set of weights; the second
                is the proposed shift vector assuming step size 1.

        Returns:
            xprod (np.ndarray): A (num_rffs, 2) shape array, containing
                (Z^T Z + lambda**2) @ wvec.
        """
        xprod = self.lambda_**2 * wvec
        for xdata in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata)
            xprod += (xtrans.T @ (xtrans @ wvec))

        if self.verbose:
            if self.n_iter % 5 == 0:
                print(f"Nfev {self.n_iter}")
        return xprod