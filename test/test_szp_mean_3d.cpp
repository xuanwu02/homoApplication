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
    int argv_id = 1;
    std::string config(argv[argv_id++]);
    std::string data_file(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    Settings s = Settings::from_json(config);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == s.dim1 * s.dim2 * s.dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, s.eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double mean = SZp_mean_3dLorenzo(cmpData, s.dim1, s.dim2, s.dim3, decData, s.B, s.eb, intToDecmpState(stateType));
    printf("mean = %.6f\n", mean);

    double act_mean = 0;
    for(size_t i=0; i<nbEle; i++) act_mean += oriData[i];
    act_mean /= nbEle;
    printf("error = %.6f\n", fabs(act_mean - mean));

    free(decData);
    free(cmpData);

    return 0;
}
