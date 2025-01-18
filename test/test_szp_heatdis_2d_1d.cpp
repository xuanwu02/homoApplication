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
    int max_iter = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);

    using T = float;
    float src_temp = 100.0, wall_temp = 0, init_temp = 0;
    double ratio = 0.8;
    size_t nbEle = dim1 * dim2;
    size_t buffer_size = (dim1 + 2) * (dim2 + 2);
    HeatDis heatdis(src_temp, wall_temp, ratio, dim1, dim2);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    T * g = (T *)malloc(buffer_size * sizeof(T));
    T * decData = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    // initData(dim1, dim2, h, init_temp);
    heatdis.initData_noghost(h, h2, init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, cmpData, dim1, dim2, blockSideLength, errorBound, cmpSize);
    SZp_heatdis_1dLorenzo<T>(cmpData, dim1, dim2, blockSideLength, max_iter, cmpSize, src_temp, wall_temp, init_temp, ratio, errorBound, state);

    auto cmpVec = readfile<unsigned char>("h1d.dat", cmpSize);

    switch(state){
        case decmpState::postPred:
        case decmpState::prePred:{
            SZp_decompress_1dLorenzo(decData, cmpVec.data(), dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            SZp_decompress_1dLorenzo(h, cmpVec.data(), dim1+2, dim2+2, blockSideLength, errorBound);
            heatdis.trimData(h, decData);
            break;
        }
    }

    heatdis.initData(h, h2, init_temp);
    heatdis.doWork(h, h2, max_iter);
    heatdis.trimData(h, g);

    writefile("h1d.dec", decData, nbEle);
    writefile("h.dec", g, nbEle);

    double err = verify(g, decData, dim1, dim2);
    printf("max_error = %.6f\n", err);

    free(h);
    free(h2);
    free(decData);
    free(cmpData);

    return 0;
}
