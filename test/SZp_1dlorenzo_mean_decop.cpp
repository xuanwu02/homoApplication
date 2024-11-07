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
    free(h);
    // prepare cmpData
    unsigned char *cmpData = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    free(oriData);
    // dcop
    float * decData = (float *)malloc(nbEle * sizeof(float));
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    double mean = 0;
    SZp_decompress_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound, mean);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f, mean = %.14f\n", elapsed_time, mean);

    free(decData);
    free(cmpData);

    return 0;
}
