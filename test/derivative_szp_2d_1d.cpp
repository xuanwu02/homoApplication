#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_1D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    std::string data_file(argv[argv_id++]);
    size_t dim = atoi(argv[argv_id++]);
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    size_t dim3 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double eb = atof(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    if(dim == 2) dim3 = 1;

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * dx_result = (T *)malloc(nbEle * sizeof(T));
    T * dy_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_dx_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_dy_result = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress(oriData, cmpData, dim1, dim2, dim3, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZp_dxdy(cmpData, dim1, dim2, dim3, blockSideLength, eb, dx_result, dy_result, intToDecmpState(stateType));

    SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
    compute_dxdy(dim1, dim2, decData, ref_dx_result, ref_dy_result);
    double ex = 0, ey = 0, ez = 0;
    ex = verify_dxdy(ref_dx_result, dx_result, dim1, dim2);
    ey = verify_dxdy(ref_dy_result, dy_result, dim1, dim2);
    printf("max error = (%.6e, %.6e, %.6e)\n", ex/eb, ey/eb, ez/eb);

    free(decData);
    free(cmpData);
    free(dx_result);
    free(dy_result);
    free(ref_dx_result);
    free(ref_dy_result);

    return 0;
}
