#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor3D.hpp"
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    std::string config(argv[1]);
    Settings s = Settings::from_json(config);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(s.data_file.c_str(), nbEle);
    assert(nbEle == s.dim1 * s.dim2 * s.dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, s.eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * dx_result = (T *)malloc(nbEle * sizeof(T));
    T * dy_result = (T *)malloc(nbEle * sizeof(T));
    T * dz_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dx_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dy_result = (T *)malloc(nbEle * sizeof(T));
    T * decop_dz_result = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZp_dxdydz_3dLorenzo(cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, dx_result, dy_result, dz_result, intToDecmpState(s.stateType));

    SZp_decompress_3dLorenzo(decData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb);
    compute_dxdydz(s.dim1, s.dim2, s.dim3, decData, decop_dx_result, decop_dy_result, decop_dz_result);
    double err;
    err = verify(decop_dx_result, dx_result, s.dim1, s.dim2, s.dim3);
    printf("dx max error = %.2e\n", err);
    err = verify(decop_dy_result, dy_result, s.dim1, s.dim2, s.dim3);
    printf("dy max error = %.2e\n", err);
    err = verify(decop_dz_result, dz_result, s.dim1, s.dim2, s.dim3);
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
