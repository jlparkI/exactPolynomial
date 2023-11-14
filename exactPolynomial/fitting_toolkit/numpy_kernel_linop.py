"""An operator for a single pass over the dataset."""
from scipy.sparse.linalg import LinearOperator
import numpy as np


class CPUDatasetLinop(LinearOperator):
    """Implements X^T X @ w as a linear operator.

    Attributes:
        kernel: A valid kernel object that can generate random features.
        dataset: A valid online or offline dataset object that can retrieve
            chunked data.
    """

    def __init__(self, kernel, dataset):
        """Class constructor.

        Args:
            kernel: A valid kernel object that can generate random features.
            dataset: A valid online or offline dataset object that can retrieve
                chunked data.
        """
        super().__init__(shape=(kernel.get_num_feats(),
                            kernel.get_num_feats()),
                            dtype=np.float64)
        self.kernel = kernel
        self.dataset = dataset


    def _matvec(self, x):
        """Implements the matvec for a single input vector."""
        xprod = np.zeros((x.shape[0]))
        for xdata in self.dataset.get_chunked_x_data():
            xtrans = self.kernel.transform_x(xdata)
            xprod += xtrans.T @ (xtrans @ x)
        return xprod
