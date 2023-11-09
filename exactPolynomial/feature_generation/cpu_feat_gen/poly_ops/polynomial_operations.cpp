/*!
 * # polynomial_operations.cpp
 *
 * This module generates features for an exact polynomial.
 *
 */
#include <vector>
#include <thread>
#include "polynomial_operations.h"


#define VALID_INPUTS 0
#define INVALID_INPUTS 1






/*!
 * # cpuExactQuadratic_
 *
 * Generates the features for an exact quadratic. The input
 * array is not changed and all features are written to the
 * designated output array.
 *
 * ## Args:
 *
 * + `inArray` Pointer to the first element of the input array data.
 * + `inDim0` The first dimension of inArray.
 * + `inDim1` The second dimension of inArray.
 * + `numThreads` The number of threads to use.
 */
template <typename T>
const char *cpuExactQuadratic_(T inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads){
    if (numThreads > inDim0)
        numThreads = inDim0;

    std::vector<std::thread> threads(numThreads);
    int startRow, endRow;
    int chunkSize = (inDim0 + numThreads - 1) / numThreads;

    for (int i=0; i < numThreads; i++){
        startRow = i * chunkSize;
        endRow = (i + 1) * chunkSize;
        if (endRow > inDim0)
            endRow = inDim0;
        threads[i] = std::thread(&ThreadExactQuadratic<T>, inArray, outArray,
                                startRow, endRow, inDim0, inDim1);
    }

    for (auto& th : threads)
        th.join();
    return "no_error";
}
//Instantiate functions that the wrapper will need to use.
template const char *cpuExactQuadratic_<double>(double inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);
template const char *cpuExactQuadratic_<float>(float inArray[], double *outArray,
                int inDim0, int inDim1, int numThreads);



/*!
 * # ThreadExactQuadratic
 *
 * Performs exact quadratic feature generation for one thread for a chunk of
 * the input array from startRow through endRow (each thread
 * works on its own group of rows).
 */
template <typename T>
void *ThreadExactQuadratic(T inArray[], double *outArray, int startRow,
        int endRow, int inDim0, int inDim1){
    int numInteractions = inDim1 * (inDim1 - 1) / 2;
    int outDim1 = numInteractions + 1 + 2 * inDim1;
    T *inElement = inArray + startRow * inDim1;
    double *outElement = outArray + startRow * outDim1;

    for (int i = startRow; i < endRow; i++){
        for (int j = 0; j < inDim1; j++){
            *outElement = inElement[j];
            outElement++;
            for (int k = j; k < inDim1; k++){
                *outElement = inElement[j] * inElement[k];
                outElement++;
            }
        }
        inElement += inDim1;
        outElement++;
    }
    return NULL;
}
