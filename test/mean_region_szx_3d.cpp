#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZx_3D.hpp"
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
    size_t lo1 = atoi(argv[argv_id++]);
    size_t hi1 = atoi(argv[argv_id++]);
    size_t lo2 = atoi(argv[argv_id++]);
    size_t hi2 = atoi(argv[argv_id++]);
    size_t lo3 = atoi(argv[argv_id++]);
    size_t hi3 = atoi(argv[argv_id++]);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, eb);

    lo1 = std::min(lo1, dim1);
    lo2 = std::min(lo2, dim2);
    lo3 = std::min(lo3, dim3);
    hi1 = std::min(hi1, dim1);
    hi2 = std::min(hi2, dim2);
    hi3 = std::min(hi3, dim3);
    assert(lo1 < hi1 && lo2 < hi2 && lo3 < hi3);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T *decData = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZx_compress(oriData, cmpData, dim1, dim2, dim3, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double mean = SZx_mean_region(
        cmpData, dim1, dim2, dim3,
        lo1, hi1, lo2, hi2, lo3, hi3,
        decData, blockSideLength, eb, intToDecmpState(stateType)
    );
    printf("region mean = %.6f\n", mean);

    SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
    double doc_mean = 0;
    for(size_t i=lo1; i<hi1; i++){
        for(size_t j=lo2; j<hi2; j++){
            for(size_t k=lo3; k<hi3; k++){
                doc_mean += decData[i * dim2 * dim3 + j * dim3 + k];
            }
        }
    }
    doc_mean /= ((hi1 - lo1) * (hi2 - lo2) * (hi3 - lo3));
    printf("rel error = %.6e\n", fabs((doc_mean - mean) / eb));

    free(decData);
    free(cmpData);

    return 0;
}
