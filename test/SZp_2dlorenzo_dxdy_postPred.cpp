#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZp_LorenzoPredictor2D.hpp"
#include "application_utils.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = 1800;
    size_t dim2 = 3600;
    std::string varname(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    size_t nbEle = dim1 * dim2;

    std::string filename = "/Users/xuanwu/github/datasets/CESM-ATM-cleared-1800x3600/" + varname + "_1_1800_3600.dat";
    size_t num;
    auto oriData_vec = readfile<T>(filename.c_str(), num);
    T * oriData = oriData_vec.data();
    unsigned char *cmpData = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    size_t cmpSize = 0;
    // compression
    SZp_compress_2dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    // operation
    double elapsed_time, total_time;
    struct timespec start, end;
    T * dx_result = (T *)malloc(dim1*dim2*sizeof(T));
    T * dy_result = (T *)malloc(dim1*dim2*sizeof(T));
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_dxdy_2dLorenzo_recover2PostPrediction(cmpData, dim1, dim2, blockSideLength, errorBound, dx_result, dy_result);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f\n", elapsed_time);

    // double err;
    // err = verify(decop_dx_result, dx_result, dim1, dim2);
    // printf("dx max error = %.6f\n", err);
    // err = verify(decop_dy_result, dy_result, dim1, dim2);
    // printf("dy max error = %.6f\n", err);

    free(cmpData);
    free(dx_result);
    free(dy_result);

    return 0;
}
