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

    using T = float;
    size_t dim1 = 512, dim2 = 512, dim3 = 512;
    // size_t dim1 = 100, dim2 = 500, dim3 = 500;
    double elapsed_time, total_time = 0;
    struct timespec start, end;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file_3d.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, errorBound);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    writefile("test.dat", cmpData, cmpSize);
    std::vector<unsigned char> compressed = readfile<unsigned char>("test.dat", nbEle);
    clock_gettime(CLOCK_REALTIME, &start);
    double mean = SZp_mean_3dLorenzo_recover2PrePred(compressed.data(), dim1, dim2, dim3, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);
    printf("mean = %.6f\n", mean);

    free(cmpData);

    return 0;
}
