#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZx_MeanPredictor2D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    int max_iter = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);

    using T = float;
    float src_temp = 100, wall_temp = 0, init_temp = 0;
    double ratio = 0.8;
    size_t nbEle = dim1 * dim2;
    double elapsed_time;
    struct timespec start, end;

    T * oriData = (T *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * h = (T *)malloc(nbEle * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));

    initData(dim1, dim2, oriData, init_temp);

    size_t cmpSize = 0;
    SZx_compress_2dMeanbased(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_heatdis_recover2PrePred<T>(cmpData, dim1, dim2, blockSideLength, max_iter, cmpSize, src_temp, wall_temp, ratio, errorBound, decmpState::prePred);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("  recover_time = %.6f\n", prePred_decmp_time);
    printf("  ope_cmp_time = %.6f\n", prePred_op_time);
    printf("elapsed_time = %.6f\n", elapsed_time);
    std::cout << "cmpSize = " << cmpSize << ", cr = " << (double)nbEle*1.0*sizeof(T)/cmpSize << std::endl;
    writefile("test.dat", cmpData, cmpSize);
    free(cmpData);
    size_t num;
    std::vector<unsigned char> compressed = readfile<unsigned char>("test.dat", num);
    SZx_decompress_2dMeanbased(decData, compressed.data(), dim1, dim2, blockSideLength, errorBound);
    doWork(dim1, dim2, max_iter, oriData, h, src_temp, wall_temp, ratio);
    double err = verify(oriData, decData, dim1, dim2);
    printf("max_error = %.6f\n", err);

    free(h);
    free(oriData);
    free(decData);

    return 0;
}
