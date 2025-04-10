#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZx_MeanPredictor3D.hpp"
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

    size_t cmpSize = 0;
    SZx_compress_3dMeanbased(oriData, cmpData, dim1, dim2, dim3, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double mean = SZx_mean_3d(cmpData, dim1, dim2, dim3, decData, blockSideLength, eb, intToDecmpState(stateType));
    printf("mean = %.6f\n", mean);

    // double act_mean = 0;
    // for(size_t i=0; i<nbEle; i++) act_mean += oriData[i];
    // act_mean /= nbEle;
    // printf("error = %.6f\n", fabs(act_mean - mean));

    SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
    double doc_mean = 0;
    for(size_t i=0; i<nbEle; i++) doc_mean += decData[i];
    doc_mean /= nbEle;
    printf("rel error = %.6e\n", fabs((doc_mean - mean) / doc_mean));

    free(decData);
    free(cmpData);

    return 0;
}
