#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor2D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);
    size_t dim1 = 1800;
    size_t dim2 = 3600;

    using T = float;
    double elapsed_time, total_time = 0;
    struct timespec start, end;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_2d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * decop_dx_result = (T *)malloc((dim1+1)*(dim2+1)*sizeof(T));
    T * decop_dy_result = (T *)malloc((dim1+1)*(dim2+1)*sizeof(T));
    T * dx_result = (T *)malloc(dim1*dim2*sizeof(T));
    T * dy_result = (T *)malloc(dim1*dim2*sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_2dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    clock_gettime(CLOCK_REALTIME, &start);
    SZp_dxdy_2dLorenzo(cmpData, dim1, dim2, blockSideLength, errorBound, dx_result, dy_result, decmpState::prePred);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  decompression time = %.6f\n", postPred_decmp_time);
    printf("  operation time = %.6f\n", postPred_op_time);
    printf("elapsed_time = %.6f\n", elapsed_time);

    SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    compute_dxdy(dim1, dim2, decData, decop_dx_result, decop_dy_result);
    double err;
    err = verify(decop_dx_result, dx_result, dim1, dim2);
    printf("dx max error = %.2e\n", err);
    err = verify(decop_dy_result, dy_result, dim1, dim2);
    printf("dy max error = %.2e\n", err);

    free(decData);
    free(cmpData);
    free(dx_result);
    free(dy_result);
    free(decop_dx_result);
    free(decop_dy_result);

    return 0;
}
