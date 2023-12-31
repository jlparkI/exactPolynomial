/*!
 * # hadamard_transforms.cpp
 *
 * This module performs core Hadamard transform ops.
 * It includes the following functions:
 *
 * + transformRows2D
 * Performs the unnormalized Hadamard transform on a 2d array
 */

#include <stdint.h>
#include <math.h>
#include "hadamard_transforms.h"




/*!
 * # transformRows2D
 *
 * Performs an unnormalized Hadamard transform along the last
 * dimension of an input 2d array. The transform is performed
 * in place so nothing is returned. Assumes dimensions have
 * already been checked by caller. Designed to be compatible
 * with multithreading.
 *
 * ## Args:
 *
 * + `xArray` Pointer to the first element of the array to be
 * modified. Must be a 2d array (e.g. N x C). C MUST be
 * a power of 2.
 * + `startRow` The first row to modify. When multithreading,
 * the array is split into blocks such that each thread
 * modifies its own subset of the rows.
 * + `endRow` The last row to modify.
 * + `dim1` The length of dim2 of the array (e.g. C in
 * N x C)
 */
template <typename T>
void transformRows2D(T xArray[], int startRow, int endRow,
                    int dim1){
    int idx1 = startRow;
    int i = 0, j, h = 1;
    T y;
    int rowStride = dim1;
    T *xElement, *yElement;

    //Unrolling the first few loops
    //of the transform increased speed substantially
    //(may be compiler and optimization level dependent).
    //This yields diminishing returns and does not
    //offer much improvement past 3 unrolls.
    for (idx1 = startRow; idx1 < endRow; idx1++){
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 1;
        for (i = 0; i < rowStride; i += 2){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 2;
            yElement += 2;
        }
        if (dim1 <= 2)
            continue;
        
        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 2;
	    for (i = 0; i < rowStride; i += 4){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement += 3;
            yElement += 3;
        }
        if (dim1 <= 4)
            continue;

        xElement = xArray + idx1 * rowStride;
        yElement = xElement + 4;
	    for (i = 0; i < rowStride; i += 8){
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            xElement ++;
            yElement ++;
            y = *yElement;
            *yElement = *xElement - y;
            *xElement += y;
            
            xElement += 5;
            yElement += 5;
        }
        if (dim1 <= 8)
            continue;

        //The general, non-unrolled transform.
        for (h = 8; h < dim1; h <<= 1){
            for (i = 0; i < rowStride; i += (h << 1)){
                xElement = xArray + idx1 * rowStride + i;
                yElement = xElement + h;
                for (j=0; j < h; j++){
                    y = *yElement;
                    *yElement = *xElement - y;
                    *xElement += y;
                    xElement++;
                    yElement++;
                }
            }
        }
    }
}
//Explicitly instantiate for external use.
template void transformRows2D<double>(double xArray[], int startRow, int endRow,
                    int dim1);
template void transformRows2D<float>(float xArray[], int startRow, int endRow,
                    int dim1);
