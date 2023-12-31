"""This file lists all of the currently implemented
kernels, together with the performance (the spearman's r)
expected from each on the test set included with the
module. Add to this when a new kernel is added to
ensure that kernel is included; you will need to
decide what is 'acceptable performance', likely
based on an initial experiment."""

#Each dictionary value contains 1) whether this
#is a convolution kernel and 2) the expected
#minimum performance.
IMPLEMENTED_KERNELS = {"ExactQuadratic":(False, 0.5)}
