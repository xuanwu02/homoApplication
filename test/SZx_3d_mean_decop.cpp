#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZx_MeanPredictor3D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    size_t dim1 = 512, dim2 = 512, dim3 = 512;
    // size_t dim1 = 100, dim2 = 500, dim3 = 500;
    double elapsed_time, total_time = 0;
    struct timespec start, end;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_3d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZx_compress_3dMeanbased(oriData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    clock_gettime(CLOCK_REALTIME, &start);
    SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  decompression time = %.6f\n", elapsed_time);
    total_time += elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  operation time = %.6f\n", elapsed_time);
    total_time += elapsed_time;
    printf("elapsed_time = %.6f\n", total_time);
    printf("mean = %.6f\n", mean);

    double dec_error = verify(oriData, decData, dim1, dim2);
    printf("dec_error = %.6f\n", dec_error);

    free(decData);
    free(cmpData);

    return 0;
}
