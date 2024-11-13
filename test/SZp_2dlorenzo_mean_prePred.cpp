#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZp_LorenzoPredictor2D.hpp"
#include "application_utils.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = 1800;
    size_t dim2 = 3600;
    std::string varname(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    size_t nbEle = dim1 * dim2;

    std::string filename = "/Users/xuanwu/github/datasets/CESM-ATM-cleared-1800x3600/" + varname + "_1_1800_3600.dat";
    auto oriData_vec = readfile<T>(filename.c_str(), nbEle);
    T * oriData = oriData_vec.data();
    unsigned char *cmpData = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    size_t cmpSize = 0;
    // compression
    SZp_compress_2dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);
    // mean
    double elapsed_time, total_time;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    T mean = SZp_mean_2dLorenzo_recover2PrePrediction<T>(cmpData, dim1, dim2, blockSideLength, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f\n", elapsed_time);
    printf("mean = %.6f\n", mean);
    // decompression error
    T * decData = (T *)malloc(nbEle * sizeof(T));
    SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double dec_error = verify(oriData, decData, dim1, dim2);
    printf("decompression error = %.6f\n", dec_error);

    free(decData);
    free(cmpData);

    return 0;
}
