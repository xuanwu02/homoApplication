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
    std::string config(argv[argv_id++]);
    std::string data_file(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    Settings s = Settings::from_json(config);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == s.dim1 * s.dim2);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, s.eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * dx_result = (T *)malloc(nbEle * sizeof(T));
    T * dy_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_dx_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_dy_result = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_2dLorenzo(oriData, cmpData, s.dim1, s.dim2, s.B, s.eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZp_dxdy_2dLorenzo(cmpData, s.dim1, s.dim2, s.B, s.eb, dx_result, dy_result, intToDecmpState(stateType));

    SZp_decompress_2dLorenzo(decData, cmpData, s.dim1, s.dim2, s.B, s.eb);
    compute_dxdy(s.dim1, s.dim2, decData, ref_dx_result, ref_dy_result);
    double err;
    err = verify_dxdy(ref_dx_result, dx_result, s.dim1, s.dim2);
    printf("dx max error = %.2e\n", err);
    err = verify_dxdy(ref_dy_result, dy_result, s.dim1, s.dim2);
    printf("dy max error = %.2e\n", err);

    free(decData);
    free(cmpData);
    free(dx_result);
    free(dy_result);
    free(ref_dx_result);
    free(ref_dy_result);

    return 0;
}
