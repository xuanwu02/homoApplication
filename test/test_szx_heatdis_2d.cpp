#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
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
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);

    using T = float;
    float src_temp = 100, wall_temp = 0, init_temp = 0;
    double ratio = 0.8;
    size_t nbEle = dim1 * dim2;

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * h = (T *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    T * oriData = (T *)malloc(nbEle * sizeof(T));
    initData(dim1, dim2, oriData, init_temp);
    size_t cmpSize = 0;
    SZx_compress_2dMeanbased(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);

    SZx_heatdis_2dMeanbased<T>(cmpData, dim1, dim2, blockSideLength, max_iter, cmpSize, src_temp, wall_temp, ratio, errorBound, state);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    SZx_decompress_2dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    doWork(dim1, dim2, max_iter, oriData, h, src_temp, wall_temp, ratio);
    double err = verify(oriData, decData, dim1, dim2);
    printf("max_error = %.6f\n", err);

    free(h);
    free(decData);
    free(cmpData);

    return 0;
}
