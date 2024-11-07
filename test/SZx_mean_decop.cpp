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
    // prepare cmpData & decData
    unsigned char *cmpData = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    size_t cmpSize = 0;
    SZx_compress_2dblock(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    // decop
    struct timespec start, end;
    double mean;
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_decompress_2dblock(decData, cmpData, dim1, dim2, blockSideLength, errorBound, mean);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_time1 = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.10f, mean = %.6f\n", elapsed_time1, mean);

    free(oriData);
    free(decData);
    free(cmpData);

    return 0;
}