#ifndef POLY_BASIC_CUDA_FHT_ARRAY_OPERATIONS_H
#define POLY_BASIC_CUDA_FHT_ARRAY_OPERATIONS_H



template <typename T>
const char *cudaSRHT2d(T npArray[], 
                    int8_t *radem, int dim0, int dim1);


template <typename T>
void cudaHTransform2d(T cArray[],
		int dim0, int dim1);

int getNumBlocksTransform(int arrsize, int divisor);


#endif
