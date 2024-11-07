#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZp_application_entry.hpp"
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

    // compose test data
    float * oriData = (float *)calloc(nbEle, sizeof(float));
    initData(dim1, dim2, oriData);
    float * h = (float *)calloc(nbEle, sizeof(float));
    doWork(dim1, dim2, max_iter, oriData, h);
    // prepare cmpData
    unsigned char *cmpData = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    float * decData = (float *)malloc(nbEle * sizeof(float));
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    // 
    double *decop_dx_result = (double *)malloc(nbEle * sizeof(double));
    double *decop_dy_result = (double *)malloc(nbEle * sizeof(double));
    double *homo_dx_result = (double *)malloc(nbEle * sizeof(double));
    double *homo_dy_result = (double *)malloc(nbEle * sizeof(double));
    double decop_elapsed_time, lorenzo_elapsed_time, quant_elapsed_time;
    double err_dx, err_dy;
    struct timespec start, end;
    // lorenzo
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_derivative_dec2Lorenzo_1dLorenzo(cmpData, dim1, dim2, blockSideLength, errorBound, homo_dx_result, homo_dy_result);
    clock_gettime(CLOCK_REALTIME, &end);
    lorenzo_elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f\n", lorenzo_elapsed_time);
    // decop
    SZp_decompress_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    compute_dxdy(dim1, dim2, decData, decop_dx_result, decop_dy_result);
    // verify
    err_dx = verify(decop_dx_result, homo_dx_result, dim1, dim2);
    err_dy = verify(decop_dy_result, homo_dy_result, dim1, dim2);
    printf("max_err_dx = %.3e\nmax_err_dy = %.3e\n", err_dx, err_dy);

    free(h);
    free(oriData);
    free(decData);
    free(cmpData);
    free(homo_dx_result);
    free(homo_dy_result);
    free(decop_dx_result);
    free(decop_dy_result);

    return 0;
}
