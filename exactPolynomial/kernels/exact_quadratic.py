"""Implements exact polynomial regression for a quadratic.
Note that this will become impractical if the number of
features in the input vector is large -- even 600 is on
the large side."""
import numpy as np
try:
    from cuda_poly_feats import cudaExactQuadratic, cudaInteractionsOnly
except:
    pass

from .kernel_baseclass import KernelBaseclass
from cpu_poly_feats import cpuExactQuadratic, cpuInteractionsOnly


class ExactQuadratic(KernelBaseclass):
    """An exact quadratic, not approximated, implemented as
    polynomial regression.

    Attributes:
        hyperparams (np.ndarray): This kernel has one
            hyperparameter: lambda_ (noise).
        poly_func: A reference to the Cython-wrapped C function
            that will be used for feature generation.
        interactions_only (bool): If True, all x**2 features are omitted.
    """

    def __init__(self, xdim, device = "cpu", num_threads = 2, interactions_only = False):
        """Constructor.

        Args:
            xdim (tuple): The dimensions of the input. Only 3d arrays are
                accepted, where shape[1] is the number of vertices in a graph
                and shape[2] is the number of features per vertex. For a fixed
                vector input, shape[1] can be 1.
            device (str): One of 'cpu', 'gpu'. Indicates the starting device.
            num_threads (int): The number of threads to use if running on CPU. If
                running on GPU, this is ignored.
            interactions_only (bool): If True, all x**2 features are omitted.
        """
        if interactions_only:
            actual_num_feats = 1 + xdim[1] + int((xdim[1] * (xdim[1] - 1)) / 2)
        else:
            actual_num_feats = 1 + xdim[1] * 2 + int((xdim[1] * (xdim[1] - 1)) / 2)

        super().__init__(actual_num_feats, xdim, num_threads)

        self.hyperparams = np.ones((1))
        self.bounds = np.asarray([[1e-3,1e1]])

        self.poly_func = None
        self.interactions_only = interactions_only
        self.device = device


    def kernel_specific_set_device(self, new_device):
        """Called when device is changed. Moves
        some of the object parameters to the appropriate device."""
        if new_device == "gpu":
            if self.interactions_only:
                self.poly_func = cudaInteractionsOnly
            else:
                self.poly_func = cudaExactQuadratic
        else:
            if self.interactions_only:
                self.poly_func = cpuInteractionsOnly
            else:
                self.poly_func = cpuExactQuadratic



    def transform_x(self, input_x):
        """Generates random features.

        Args:
            input_x: A cupy or numpy array depending on self.device
                containing the input data.

        Returns:
            output_x: A cupy or numpy array depending on self.device
                containing the results of random feature generation. Note
                that num_feats rffs are generated, not num_feats.
        """
        if len(input_x.shape) != 2:
            raise ValueError("Input to ClassicPoly must be a 2d array.")
        retyped_input = input_x.astype(self.dtype)
        output_x = self.zero_arr((input_x.shape[0], self.num_feats),
                self.out_type)
        output_x[:,-1] = 1

        self.poly_func(retyped_input, output_x, self.num_threads)
        return output_x


    def kernel_specific_set_hyperparams(self):
        """Provided for consistency with baseclass. This
        kernel has no kernel-specific properties that must
        be reset after hyperparameters are changed."""
        return
