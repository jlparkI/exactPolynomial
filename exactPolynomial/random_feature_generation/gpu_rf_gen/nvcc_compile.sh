#!/bin/bash

nvcc --compiler-options '-fPIC'  \
    -c -o poly_arrop_temp.o poly_ops/polynomial_operations.cu

ar cru libarray_operations.a poly_arrop_temp.o


ranlib libarray_operations.a

rm -f poly_arrop_temp.o
