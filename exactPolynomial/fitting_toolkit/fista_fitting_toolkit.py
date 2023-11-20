"""Contains the tools needed to fit using the FISTA algorithm."""
import numpy as np
try:
    import cupy as cp
    from .cupy_kernel_linop import CudaDatasetLinop
    from cupyx.scipy.sparse.linalg import eigsh as Cuda_eigsh
except:
    pass
from scipy.sparse.linalg import eigsh as CPU_eigsh
from .numpy_kernel_linop import CPUDatasetLinop



class FISTAFit:
    """This class contains all the tools needed to fit a model
    whose hyperparameters have already been tuned using FISTA.

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
        soft_thresh: A convenience reference to a device-specific thresholding
            function.
        signval: A convenience reference to cp.sign or np.sign.
        niter (int): The number of function evaluations performed.
        lipschitz (float): The largest eigenvalue.
    """

    def __init__(self, dataset, kernel, device, verbose):
        """Class constructor.

        Args:
            dataset: An OnlineDataset or OfflineDatset containing all the
                training data.
            kernel: A kernel object that can generate random features for
                the Dataset.
            device (str): One of 'cpu', 'gpu'. Indicates where calculations
                will be performed.
            verbose (bool): If True, print regular updates.
        """
        self.dataset = dataset
        self.kernel = kernel
        self.lambda_ = kernel.get_lambda()
        self.verbose = verbose
        self.device = device
        if device == "gpu":
            self.zero_arr = cp.zeros
            self.dtype = cp.float64
            self.soft_thresh = self.cuda_soft_thresh
            linop = CudaDatasetLinop(kernel, dataset)
            self.lipschitz = Cuda_eigsh(linop, k=1, return_eigenvectors=False)[0]

        else:
            self.zero_arr = np.zeros
            self.dtype = np.float64
            self.soft_thresh = self.cpu_soft_thresh
            linop = CPUDatasetLinop(kernel, dataset)
            self.lipschitz = CPU_eigsh(linop, k=1, return_eigenvectors=False)[0]
        self.n_iter = 0



    def fit_model(self, max_iter = 500, tol = 1e-6):
        """Finds an optimal set of weights using the information already
        provided to the class constructor.

        Args:
            max_iter (int): The maximum number of iterations.
            tol (float): The threshold for convergence.

        Returns:
            wvec: A cupy or numpy array depending on device that contains the
                best set of weights found. A 1d array of length self.kernel.get_num_feats().
        """
        self.n_iter = 0
        x_k = self.zero_arr((self.kernel.get_num_feats()))
        y_k = x_k.copy()
        t_k = 1

        while self.n_iter < max_iter:
            old_y, old_x = y_k.copy(), x_k.copy()
            x_k = self.soft_thresh(y_k - self.full_dataset_pass(y_k))

            t_k_1 = (1. + np.sqrt(1. + 4. * t_k**2)) / 2.
            y_k = x_k + ((t_k - 1.) / t_k_1) * (x_k - old_x)
            t_k = t_k_1

            wvec_shift = x_k - old_x
            wvec_shift = np.sqrt(float(wvec_shift.T @ wvec_shift))
            wvec_shift /= np.sqrt(float(x_k.T @ x_k))

            if wvec_shift < tol:
                break
            if self.n_iter % 5 == 0:
                print(f"Iteration {self.n_iter}, wvec shift {wvec_shift}")
            self.n_iter += 1

        if self.device == "gpu":
            x_k = cp.asarray(x_k)
        return x_k, self.n_iter, []


    def cuda_soft_thresh(self, wvec):
        """Soft thresholding for cupy only."""
        return cp.sign(wvec) * cp.maximum(cp.abs(wvec) - (self.lambda_**2 / self.lipschitz), 0.)

    def cpu_soft_thresh(self, wvec):
        """Soft thresholding for numpy only."""
        return np.sign(wvec) * np.maximum(np.abs(wvec) - (self.lambda_**2 / self.lipschitz), 0.)



    def full_dataset_pass(self, wvec):
        """Runs a full pass over the dataset."""
        xprod = self.zero_arr((wvec.shape[0]))
        for xchunk, ychunk in self.dataset.get_chunked_data():
            xtrans = self.kernel.transform_x(xchunk)
            xprod += xtrans.T @ (xtrans @ wvec - ychunk)
        return xprod / self.lipschitz
