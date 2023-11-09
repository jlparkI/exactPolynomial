/*!
 * # diagonal_matmul_ops.cpp
 *
 * This module performs core diagonal matrix
 * multiplication operations.
 * It includes the following functions:
 *
 * + multiplyByDiagonalRademacherMat2d
 * Multiplies a 2d array by a diagonal matrix whose elements
 * are drawn from a Rademacher distribution
 */

#include <math.h>
#include "diagonal_matmul_ops.h"



/*!
 * # multiplyByDiagonalRademacherMat2D
 *
 * Multiplies an input 2d array xArray by a 1d array rademArray assumed
 * to represent a diagonal matrix. rademArray should
 * therefore be of shape (C) if xArray is of shape (N, C).
 * Thus each element (i, j) of xArray is multiplied by
 * element (j) of rademArray. Function assumes caller has
 * verified all dimensions. The array is also multiplied by the normalization
 * constant for the Hadamard transform.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 2d array (e.g. N x C)
 * + `rademArray` A 1d array to multiply against xArray
 * of shape (C)
 * + `dim1` The length of dim2 of xArray (e.g. C in
 * N x C)
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 *
 * ## Returns:
 * Operations are in place so nothing is returned.
 */
template <typename T>
void multiplyByDiagonalRademacherMat2D(T xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow){
    
    int i = startRow, j = i;
    T normConstant = log2(dim1) / 2;
    normConstant = 1 / pow(2, normConstant);
    int rowStride = dim1;
    T *xElement;
    
    for(i = startRow; i < endRow; i++){
        xElement = xArray + i * rowStride;
        for (j = 0; j < rowStride; j++){
            *xElement *= rademArray[j] * normConstant;
            xElement++;
        }
    }
}
//Explicitly instantiate for external use.
template void multiplyByDiagonalRademacherMat2D<float>(float xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);
template void multiplyByDiagonalRademacherMat2D<double>(double xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);
