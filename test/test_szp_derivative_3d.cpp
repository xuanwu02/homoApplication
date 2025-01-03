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
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);

    using T = float;
    size_t dim1 = 512, dim2 = 512, dim3 = 512;
    // size_t dim1 = 100, dim2 = 500, dim3 = 500;
    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_3d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * dx_result = (T *)malloc(nbEle * sizeof(T));
    T * dy_result = (T *)malloc(nbEle * sizeof(T));
    T * dz_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dx_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dy_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dz_result = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZp_dxdydz_3dLorenzo(cmpData, dim1, dim2, dim3, blockSideLength, errorBound, dx_result, dy_result, dz_result, state);

    SZp_decompress_3dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    compute_dxdydz(dim1, dim2, dim3, decData, decop_dx_result, decop_dy_result, decop_dz_result);
    double err;
    err = verify(decop_dx_result, dx_result, dim1, dim2);
    printf("dx max error = %.2e\n", err);
    err = verify(decop_dy_result, dy_result, dim1, dim2);
    printf("dy max error = %.2e\n", err);
    err = verify(decop_dz_result, dz_result, dim1, dim2, dim3);
    printf("dz max error = %.2e\n", err);

    free(decData);
    free(cmpData);
    free(dx_result);
    free(dy_result);
    free(dz_result);
    free(decop_dx_result);
    free(decop_dy_result);
    free(decop_dz_result);

    return 0;
}
