#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor3D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    size_t dim1 = 100, dim2 = 500, dim3 = 500;
    double elapsed_time, total_time = 0;
    struct timespec start, end;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_3d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * decop_dx_result = (T *)malloc(nbEle*sizeof(T));
    T * decop_dy_result = (T *)malloc(nbEle*sizeof(T));
    T * decop_dz_result = (T *)malloc(nbEle*sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_decompress_3dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  decompression time = %.6f\n", elapsed_time);
    total_time += elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    compute_dxdydz(dim1, dim2, dim3, decData, decop_dx_result, decop_dy_result, decop_dz_result);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  operation time = %.6f\n", elapsed_time);
    total_time += elapsed_time;
    printf("elapsed_time = %.6f\n", total_time);

    double dec_error = verify(oriData, decData, dim1, dim2, dim3);
    printf("dec_error = %.6f\n", dec_error);

    free(decData);
    free(cmpData);
    free(decop_dx_result);
    free(decop_dy_result);
    free(decop_dz_result);

    return 0;
}
