#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZx_application_entry.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    if((dim1 % blockSideLength) || (dim2 % blockSideLength)){
        printf("incompatible blockSideLength\n");
        exit(0);
    }
    double errorBound = atof(argv[argv_id++]);
    int max_iter = atoi(argv[argv_id++]);
    size_t nbEle = dim1 * dim2;

    using T = float;
    // compose test data
    T * oriData = (T *)calloc(nbEle, sizeof(T));
    initData(dim1, dim2, oriData);
    T * h = (T *)calloc(nbEle, sizeof(T));
    doWork(dim1, dim2, max_iter, oriData, h);
    free(h);
    // prepare cmpData
    unsigned char *cmpData = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    size_t cmpSize = 0;
    SZx_compress_2dblock(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    free(oriData);
    // diff
    double *homo_dx_result = (double *)malloc(nbEle * sizeof(double));
    double *homo_dy_result = (double *)malloc(nbEle * sizeof(double));
    double elapsed_time;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_derivative_diff_2dblock(cmpData, dim1, dim2, blockSideLength, homo_dx_result, homo_dy_result, errorBound, elapsed_time);
    printf("elapsed_time = %.10f\n", elapsed_time);

    free(cmpData);
    free(homo_dx_result);
    free(homo_dy_result);

    return 0;
}