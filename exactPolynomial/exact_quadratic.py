"""Describes the ExactQuadratic class for fitting an exact
quadratic to a (potentially) large dataset.
"""
import sys
try:
    import cupy as cp
except:
    print("CuPy not detected. xGPR will run in CPU-only mode.")
import numpy as np
from .kernels import KERNEL_NAME_TO_CLASS

from .fitting_toolkit.lbfgs_fitting_toolkit import lBFGSModelFit



class ExactQuadratic():
    """A class for fitting an exact quadratic to a (potentially)
    large dataset, using L1 or L2 regularization."""

    def __init__(self, device = "cpu", verbose = True,
                    num_threads = 2):
        """Class constructor.

        Args:
            device (str): Determines whether calculations are performed on
                'cpu' or 'gpu'. The initial entry can be changed later
                (i.e. model can be transferred to a different device).
                Defaults to 'cpu'.
            verbose (bool): If True, regular updates are printed
                during fitting and tuning. Defaults to True.
            num_threads (int): The number of threads to use for random feature generation
                if running on CPU. If running on GPU, this argument is ignored.
        """
        self.kernel = None
        self.weights = None
        self.device = device

        self.num_threads = num_threads

        self.verbose = verbose
        self.trainy_mean = 0.0
        self.trainy_std = 1.0

        self.kernel_spec_parms = {}


    def initialize(self, dataset, random_seed = 123, hyperparams = None, input_bounds = None):
        """Initializes the kernel using the supplied dataset and random seed.
        If the kernel has already been initialized, no further action is
        taken.

        Args:
            dataset: An Online or Offline dataset for training.
            random_seed (int): A random seed to set up the kernel.
            hyperparams (ndarray): Either None or a numpy array. If not None,
                must be a numpy array such that shape[0] == the number of hyperparameters
                for the selected kernel. The kernel hyperparameters are then initialized
                to the specified value. If None, default hyperparameters are used which
                should then be tuned.
            input_bounds (np.ndarray): The bounds for optimization. Defaults to
                None, in which case the kernel uses its default bounds.
                If supplied, must be a 2d numpy array of shape (num_hyperparams, 2).

        Raises:
            ValueError: A ValueError is raised if invalid inputs are supplied.
        """
        self.kernel = self._initialize_kernel("ExactQuadratic", dataset.get_xdim(),
                        random_seed, input_bounds)
        self.weights = None
        if hyperparams is not None:
            self.kernel.check_hyperparams(hyperparams)
            self.kernel.set_hyperparams(hyperparams, logspace = True)


    def pre_prediction_checks(self, input_x):
        """Checks input data to ensure validity.

        Args:
            input_x (np.ndarray): A numpy array containing the input data.

        Returns:
            x_array: A cupy array (if self.device is gpu) or a reference
                to the unmodified input array otherwise.

        Raises:
            ValueError: If invalid inputs are supplied,
                a detailed ValueError is raised to explain.
        """
        x_array = input_x
        if self.kernel is None or self.weights is None:
            raise ValueError("Model has not yet been successfully fitted.")
        if not self.kernel.validate_new_datapoints(input_x):
            raise ValueError("The input has incorrect dimensionality.")
        #This should never happen, but just in case.
        if self.weights.shape[0] != self.kernel.get_num_feats():
            raise ValueError("The size of the weight vector does not "
                    "match the number of random features that are generated.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            x_array = cp.asarray(input_x)

        return x_array


    def get_hyperparams(self):
        """Simple helper function to return hyperparameters if the model
        has already been tuned or fitted."""
        if self.kernel is None:
            return None
        return self.kernel.get_hyperparams()



    def _initialize_kernel(self, kernel_choice, input_dims,
                        random_seed, bounds = None):
        """Selects and initializes an appropriate kernel object based on the
        kernel_choice string supplied by caller. The kernel is then moved to
        the appropriate device based on the 'device' supplied by caller
        and is returned.

        Args:
            kernel_choice (str): The kernel selection. Must be one of
                constants.ACCEPTABLE_KERNELS.
            input_dims (list): The dimensions of the data. A list of
                [ndatapoints, x.shape[1]] for non-convolution data
                or [ndatapoints, x.shape[1], x.shape[2]] for conv1d
                data.
            random_seed (int): The random seed to the random number
                generator the kernel uses to initialize.
            bounds (np.ndarray): The bounds on hyperparameter
                tuning. Must have an appropriate shape for the
                selected kernel. If None, the kernel will use
                its defaults. Defaults to None.

        Returns:
            kernel: An object of the appropriate kernel class.

        Raises:
            ValueError: Raises a value error if an unrecognized kernel
                is supplied.
        """
        if kernel_choice not in KERNEL_NAME_TO_CLASS:
            raise ValueError("An unrecognized kernel choice was supplied.")
        kernel = KERNEL_NAME_TO_CLASS[kernel_choice](input_dims,
                            random_seed, self.device, self.num_threads,
                            kernel_spec_parms = self.kernel_spec_parms)
        if bounds is not None:
            kernel.set_bounds(bounds)
        return kernel


    def predict(self, input_x, chunk_size = 2000):
        """Generate a predicted value for each
        input datapoint -- and if desired the variance.

        Args:
            input_x (np.ndarray): The input data. Should be a 2d numpy
                array (if non-convolution kernel) or 3d (if convolution
                kernel).
            chunk_size (int): The number of datapoints to process at
                a time. Lower values limit memory consumption. Defaults
                to 2000.

        Returns:
            Returns predictions, a numpy array of length N
            for N datapoints.

        Raises:
            ValueError: If the dimesionality or type of the input does
                not match what is expected, or if the model has
                not yet been fitted, a ValueError is raised.
        """
        xdata = self.pre_prediction_checks(input_x)
        preds = []

        for i in range(0, xdata.shape[0], chunk_size):
            cutoff = min(i + chunk_size, xdata.shape[0])
            xfeatures = self.kernel.transform_x(xdata[i:cutoff, :])
            preds.append((xfeatures * self.weights[None, :]).sum(axis = 1))

        if self.device == "gpu":
            preds = cp.asnumpy(cp.concatenate(preds))
        else:
            preds = np.concatenate(preds)
        return preds * self.trainy_std + self.trainy_mean



    def _run_pre_fitting_prep(self, dataset, preset_hyperparams = None):
        """Runs key steps / checks needed if about to fit the
        model.
        """
        dataset.device = self.device
        self.trainy_mean = dataset.get_ymean()
        self.trainy_std = dataset.get_ystd()

        if self.kernel is None:
            raise ValueError("Must call self.initialize before fitting.")

        if preset_hyperparams is not None:
            self.kernel.check_hyperparams(preset_hyperparams)
            self.kernel.set_hyperparams(preset_hyperparams, logspace = True)


    def _run_post_fitting_cleanup(self, dataset):
        """Runs key steps / checks needed if just finished
        fitting the model.
        """
        dataset.device = "cpu"


    def fit(self, dataset, regularization = "l1", tol = 1e-6,
                preset_hyperparams=None, max_iter = 500,
                run_diagnostics = False, mode = "lbfgs"):
        """Fits the model after checking that the input data
        is consistent with the kernel choice and other user selections.

        Args:
            dataset: Object of class OnlineDataset or OfflineDataset.
            regularization (str): One of 'l1', 'l2'. Determines the type of
                regularization which is applied.
            tol (float): The threshold below which iterative strategies (L-BFGS, CG,
                SGD) are deemed to have converged. Defaults to 1e-5. Note that how
                reaching the threshold is assessed may depend on the algorithm.
            preset_hyperparams: Either None or a numpy array. If None,
                hyperparameters must already have been tuned using one
                of the tuning methods (e.g. tune_hyperparams_bayes_bfgs).
                If supplied, must be a numpy array of shape (N, 2) where
                N is the number of hyperparams for the kernel in question.
            max_iter (int): The maximum number of epochs for iterative strategies.
            random_seed (int): The random seed for the random number generator.
            run_diagnostics (bool): If True, the number of conjugate
                gradients and the preconditioner diagnostics ratio are returned.
            mode (str): Must be one of "cg", "lbfgs", "exact".
                Determines the approach used. If 'exact', self.kernel.get_num_feats
                must be <= constants.constants.MAX_CLOSED_FORM_RFFS.
            suppress_var (bool): If True, do not calculate variance. This is generally only
                useful when optimizing hyperparameters, since otherwise we want to calculate
                the variance. It is best to leave this as default False unless performing
                hyperparameter optimization.

        Returns:
            Does not return anything unless run_diagnostics is True.
            n_iter (int): The number of iterations if applicable.
            losses (list): The loss on each iteration. Only for SGD and CG, otherwise,
                empty list.

        Raises:
            ValueError: The input dataset is checked for validity before tuning is
                initiated, an error is raised if problems are found."""
        if regularization not in ['l1', 'l2']:
            raise ValueError("Unrecognized regularization option supplied.")
        self._run_pre_fitting_prep(dataset, preset_hyperparams)
        self.weights = None

        if self.verbose:
            print("starting fitting")
        if mode == "lbfgs":
            model_fitter = lBFGSModelFit(dataset, regularization, self.kernel,
                    self.device, self.verbose)
            self.weights, n_iter, losses = model_fitter.fit_model_lbfgs(max_iter, tol)

        else:
            raise ValueError("Unrecognized fitting mode supplied. Must provide one of "
                        "'lbfgs'.")

        if self.verbose:
            print("Fitting complete.")
        if self.device == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._run_post_fitting_cleanup(dataset)

        if run_diagnostics:
            return n_iter, losses


    ####The remaining functions are all getters / setters.


    @property
    def kernel_spec_parms(self):
        """Property definition for the kernel_spec_parms."""
        return self._kernel_spec_parms

    @kernel_spec_parms.setter
    def kernel_spec_parms(self, value):
        """Setter for kernel_spec_parms. If the
        user is changing this, the kernel needs to be
        re-initialized."""
        if not isinstance(value, dict):
            raise ValueError("Tried to set kernel_spec_parms to something that "
                    "was not a dict!")
        self._kernel_spec_parms = value
        self.kernel = None
        self.weights = None


    @property
    def num_threads(self):
        """Property definition for the num_threads attribute."""
        return self._num_threads

    @num_threads.setter
    def num_threads(self, value):
        """Setter for the num_threads attribute."""
        if value > 24 or value < 1:
            self._num_threads = 2
            raise ValueError("Num threads if supplied must be an integer from 1 to 24.")
        self._num_threads = value
        if self.kernel is not None:
            self.kernel.num_threads = value


    @property
    def device(self):
        """Property definition for the device attribute."""
        return self._device

    @device.setter
    def device(self, value):
        """Setter for the device attribute."""
        if value not in ["cpu", "gpu"]:
            raise ValueError("Device must be in ['cpu', 'gpu'].")

        if "cupy" not in sys.modules and value == "gpu":
            raise ValueError("You have specified the gpu fit mode but CuPy is "
                "not installed. Currently CPU only fitting is available.")

        if "cuda_rf_gen_module" not in sys.modules and value == "gpu":
            raise ValueError("You have specified the gpu fit mode but the "
                "cudaHadamardTransform module is not installed / "
                "does not appear to have installed correctly. "
                "Currently CPU only fitting is available.")

        if self.kernel is not None:
            self.kernel.device = value
        if self.weights is not None:
            if value == "gpu":
                self.weights = cp.asarray(self.weights)
            elif value == "cpu" and not isinstance(self.weights, np.ndarray):
                self.weights = cp.asnumpy(self.weights)
        if value == "gpu":
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        self._device = value
