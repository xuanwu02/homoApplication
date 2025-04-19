#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_3D.hpp"
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

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * laplacian_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_laplacian_result = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress(oriData, cmpData, dim1, dim2, dim3, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZp_laplacian(cmpData, dim1, dim2, dim3, blockSideLength, eb, laplacian_result, intToDecmpState(stateType));

    SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
    compute_laplacian_3d(dim1, dim2, dim3, decData, ref_laplacian_result);
    double err;
    err = verify_dxdydz(ref_laplacian_result, laplacian_result, dim1, dim2, dim3);
    printf("max error = (%.6e, 0, 0)\n", err/eb);

    free(decData);
    free(cmpData);
    free(laplacian_result);
    free(ref_laplacian_result);

    return 0;
}
