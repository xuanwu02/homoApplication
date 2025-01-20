#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor1D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);
    int max_iter = atoi(argv[argv_id++]);
    ht_plot_gap = 1;

    size_t nbEle = dim1 * dim2;
    size_t buffer_size = (dim1 + 2) * (dim2 + 2);

    float src_temp = 100.0;
    float wall_temp = 0.0;
    float init_temp = 0.0;
    float ratio = 0.8;
    HeatDis heatdis(src_temp, wall_temp, ratio, dim1, dim2);

    using T = float;

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    heatdis.initData_noghost(h, h2, init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    SZp_heatdis_1dLorenzo<T>(cmpData, dim1, dim2, blockSideLength, max_iter, cmpSize, src_temp, wall_temp, init_temp, ratio, errorBound, state, false);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
