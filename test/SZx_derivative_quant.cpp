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
    SZx_compress_2dblock(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    // quant
    T *homo_dx_result = (T *)malloc(nbEle * sizeof(T));
    T *homo_dy_result = (T *)malloc(nbEle * sizeof(T));
    double elapsed_time;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_derivative_quant_2dblock(cmpData, dim1, dim2, blockSideLength, homo_dx_result, homo_dy_result, errorBound, elapsed_time);
    printf("elapsed_time = %.10f\n", elapsed_time);
    // verify
    T *decop_dx_result = (T *)malloc(nbEle * sizeof(T));
    T *decop_dy_result = (T *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    SZx_decompress_2dblock(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    compute_dxdy(dim1, dim2, decData, decop_dx_result, decop_dy_result);
    double err_dx, err_dy;
    err_dx = verify(decop_dx_result, homo_dx_result, dim1, dim2);
    err_dy = verify(decop_dy_result, homo_dy_result, dim1, dim2);
    printf("max_err_dx = %.3e\nmax_err_dy = %.3e\n", err_dx, err_dy);

    free(decData);
    free(cmpData);
    free(homo_dx_result);
    free(homo_dy_result);
    free(decop_dx_result);
    free(decop_dy_result);

    return 0;
}