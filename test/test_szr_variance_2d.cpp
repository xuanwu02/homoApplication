#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZr_RegressionPredictor2D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);

    using T = float;
    size_t dim1 = 1800, dim2 = 3600;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_2d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZr_compress_2dRegression(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double var = SZr_variance_2d(cmpData, dim1, dim2, decData, blockSideLength, errorBound, state);
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
