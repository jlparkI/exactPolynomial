"""Wraps the C functions that generate random features for poly &
graph poly kernels.

Also performs all of the bounds and safety checks needed to use these
functions (the C functions do not do their own bounds checking). It
is EXTREMELY important that this wrapper not be bypassed for this
reason -- it double checks all of the array dimensions, types,
is data contiguous etc. before calling the wrapped C functions."""
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from libc cimport stdint
from libc.stdint cimport int8_t, int32_t
import math



cdef extern from "poly_ops/polynomial_operations.h" nogil:
    const char *cpuExactQuadratic_[T](T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads)



@cython.boundscheck(False)
@cython.wraparound(False)
def cpuExactQuadratic(np.ndarray[floating, ndim=2] inputArray,
                np.ndarray[double, ndim=2] outputArray,
                int numThreads):
    """Wraps C++ operations for generating features for an exact
    quadratic.

    Args:
        inputArray (np.ndarray): The input data. This is not modified.
        outputArray (np.ndarray): The output array. Must have the appropriate
            shape such that all of the quadratic polynomial features can
            be written to it. The last column is assumed to be saved for 1
            for a y-intercept term.
        num_threads (int): Number of threads.

    Raises:
        ValueError: A ValueError is raised if unexpected or invalid inputs are supplied.
    """
    cdef const char *errCode
    cdef uintptr_t addr_input = inputArray.ctypes.data
    cdef int numExpectedFeats = int( inputArray.shape[1] * (inputArray.shape[1] - 1) / 2)
    numExpectedFeats += 2 * inputArray.shape[1] + 1

    if inputArray.shape[0] == 0:
        raise ValueError("There must be at least one datapoint.")
    if inputArray.shape[0] != outputArray.shape[0]:
        raise ValueError("The number of datapoints in the outputs and the inputs do "
                "not agree.")
    if outputArray.shape[1] != numExpectedFeats:
        raise ValueError("The shape of the output array is incorrect for a quadratic.")

    if not outputArray.flags["C_CONTIGUOUS"] or not inputArray.flags["C_CONTIGUOUS"]:
        raise ValueError("One or more arguments is not C contiguous.")

    if inputArray.dtype == "float32":
        errCode = cpuExactQuadratic_[float](<float*>addr_input, &outputArray[0,0],
                        inputArray.shape[0], inputArray.shape[1], numThreads)

    elif inputArray.dtype == "float64":
        errCode = cpuExactQuadratic_[double](<double*>addr_input, &outputArray[0,0],
                        inputArray.shape[0], inputArray.shape[1], numThreads)

    else:
        raise ValueError("Unexpected types passed to wrapped C++ function.")

    if errCode.decode("UTF-8") != "no_error":
        raise Exception("Fatal error encountered while performing graph convolution.")
