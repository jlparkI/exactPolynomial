#ifndef POLY_TRANSFORM_FUNCTIONS_H
#define POLY_TRANSFORM_FUNCTIONS_H



template <typename T>
const char *SRHTBlockTransform_(T Z[], int8_t *radem,
            int zDim0, int zDim1, int numThreads);

template <typename T>
void *ThreadSRHTRows2D(T arrayStart[], int8_t* rademArray,
        int dim1, int startPosition, int endPosition);

#endif
