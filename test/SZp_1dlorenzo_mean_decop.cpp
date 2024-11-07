#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZp_application_entry.hpp"
#include "application_utils.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = 1800;
    size_t dim2 = 3600;
    int blockSideLength = atoi(argv[argv_id++]);
    if((dim1 % blockSideLength) || (dim2 % blockSideLength)){
        printf("incompatible blockSideLength\n");
        exit(0);
    }
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    size_t nbEle;
    auto oriData_vec = readfile<T>(cldhigh_data_file.c_str(), nbEle);
    T * oriData = oriData_vec.data();
    // prepare cmpData
    unsigned char *cmpData = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    // decop
    T * decData = (T *)malloc(nbEle * sizeof(T));
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_decompress_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("decompression time = %.6f\n", elapsed_time);
    clock_gettime(CLOCK_REALTIME, &start);
    double mean = 0;
    for(int i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("opreation time = %.6f, mean = %.14f\n", elapsed_time, mean);

    double dec_error = verify(oriData, decData, dim1, dim2);
    printf("decompression error = %.6f\n", dec_error);

    free(decData);
    free(cmpData);

    return 0;
}
