/*!
 * # transform_functions.c
 *
 * This module uses the "low-level" functions in array_operations to perform
 * SRHT operations.
 *
 * + SRHTBlockTransform_
 * Performs the key operations in the SRHT on an input 2d array using multithreading.
 *
 * + ThreadSRHTRows
 * Performs operations for a single thread of the SRHT operation.
 */
#include <Python.h>
#include <vector>
#include <thread>
#include "transform_functions.h"
#include "../shared_fht_functions/hadamard_transforms.h"
#include "../shared_fht_functions/diagonal_matmul_ops.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1


/*!
 * # SRHTBlockTransform_
 *
 * Performs the following operation along the last dimension
 * of the input array Z: H D1, where H is a normalized
 * Hadamard transform and D1 is a diagonal array.
 *
 * ## Args:
 *
 * + `Z` Pointer to the first element of the array to be
 * modified. Must be a 2d array (N x C). C must be a power of 2.
 * + `radem` A diagonal array of shape (C).
 * + `zDim0` The first dimension of Z (N).
 * + `zDim1` The second dimension of Z (C)
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *SRHTBlockTransform_(T Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads)
{
    if (numThreads > zDim0)
        numThreads = zDim0;

    std::vector<std::thread> threads(numThreads);
    int startPosition, endPosition;
    int chunkSize = (zDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startPosition = i * chunkSize;
        endPosition = (i + 1) * chunkSize;
        if (endPosition > zDim0)
            endPosition = zDim0;
        threads[i] = std::thread(&ThreadSRHTRows2D<T>, Z, radem,
                zDim1, startPosition, endPosition);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}


/*!
 * # ThreadSRHTRows2D
 *
 * Performs the SRHT operation for one thread for a chunk of
 * the input array Z from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadSRHTRows2D(T arrayStart[], int8_t* rademArray,
        int dim1, int startPosition, int endPosition){

    multiplyByDiagonalRademacherMat2D<T>(arrayStart,
                    rademArray, dim1,
                    startPosition, endPosition);
    transformRows2D<T>(arrayStart, startPosition, 
                    endPosition, dim1);
    return NULL;
}

template const char *SRHTBlockTransform_<float>(float Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);
template const char *SRHTBlockTransform_<double>(double Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);
