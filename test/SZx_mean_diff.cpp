#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZx_application_entry.hpp"
#include "application_utils.hpp"
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
    size_t nbEle = dim1 * dim2;

    using T = float;
    // compose oriData
    T * oriData = (T *)malloc(nbEle * sizeof(T));
    T min = -100, max = 100;
    int seed = 24;
    initRandomData(min, max, seed, nbEle, oriData);
    // prepare cmpData
    unsigned char *cmpData = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    size_t cmpSize = 0;
    SZx_compress_2dblock(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    free(oriData);
    // homo
    T * decData = (T *)malloc(nbEle * sizeof(T));
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    double mean = SZx_mean_2dblock(cmpData, dim1, dim2, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_time2 = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.10f, mean = %.16f\n", elapsed_time2, mean);

    free(decData);
    free(cmpData);

    return 0;
}