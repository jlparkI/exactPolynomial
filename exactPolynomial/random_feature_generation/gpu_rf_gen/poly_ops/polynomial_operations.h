#ifndef CUDA_POLYNOMIAL_OPERATIONS_H
#define CUDA_POLYNOMIAL_OPERATIONS_H
#include <stdint.h>

template <typename T>
const char *cudaExactQuadratic_(T inArray[], double *outArray, 
                    int inDim0, int inDim1);

#endif
