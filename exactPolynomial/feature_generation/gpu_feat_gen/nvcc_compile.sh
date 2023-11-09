#!/bin/bash

nvcc --compiler-options '-fPIC'  \
    -c -o poly_arrop_temp.o poly_ops/polynomial_operations.cu
nvcc --compiler-options '-fPIC'  \
    -c -o basic_op_temp.o basic_ops/basic_array_operations.cu

ar cru libarray_operations.a basic_op_temp.o poly_arrop_temp.o


ranlib libarray_operations.a

rm -f poly_arrop_temp.o basic_op_temp.o
