#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZr_RegressionPredictor3D.hpp"
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

    size_t cmpSize = 0;
    SZr_compress_3dRegression(oriData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double var = SZr_variance_3d(cmpData, s.dim1, s.dim2, s.dim3, decData, s.B, s.eb, intToDecmpState(s.stateType));
    printf("variance = %.6f\n", var);

    double act_mean = 0;
    for(size_t i=0; i<nbEle; i++) act_mean += oriData[i];
    act_mean /= nbEle;
    double act_var = 0;
    for(size_t i=0; i<nbEle; i++) act_var += (oriData[i] - act_mean) * (oriData[i] - act_mean);
    act_var /= (nbEle - 1);
    printf("error = %.6f\n", fabs(act_var - var));

    free(decData);
    free(cmpData);

    return 0;
}
