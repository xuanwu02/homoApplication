#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor2D.hpp"
#include "utils.hpp"
#include "settings.hpp"

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
    int bufferType = atoi(argv[argv_id++]);
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
    SZp_compress_2dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    switch(bufferType){
        case 0:{
            SZp_dxdy_2dLorenzo_IntBuffer(cmpData, dim1, dim2, blockSideLength, eb, dx_result, dy_result, intToDecmpState(stateType));
            break;
        }
        case 1:{
            SZp_dxdy_2dLorenzo_FltBuffer(cmpData, dim1, dim2, blockSideLength, eb, dx_result, dy_result, intToDecmpState(stateType));
            break;
        }
        default:
            break;
    }

    SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, eb);
    compute_dxdy(dim1, dim2, decData, ref_dx_result, ref_dy_result);
    double ex = 0, ey = 0, ez = 0;
    ex = verify_dxdy(ref_dx_result, dx_result, dim1, dim2);
    ey = verify_dxdy(ref_dy_result, dy_result, dim1, dim2);
    printf("max error = (%.6e, %.6e, %.6e)\n", ex, ey, ez);

    free(decData);
    free(cmpData);
    free(dx_result);
    free(dy_result);
    free(ref_dx_result);
    free(ref_dy_result);

    return 0;
}
