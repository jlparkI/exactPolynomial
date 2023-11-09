#ifndef POLY_DIAGONAL_MATMUL_OPERATIONS_H
#define POLY_DIAGONAL_MATMUL_OPERATIONS_H
#include <stdint.h>


template <typename T>
void multiplyByDiagonalRademacherMat2D(T xArray[],
                    const int8_t *rademArray,
                    int dim1,
                    int startRow, int endRow);
#endif
