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
    size_t L = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double errorBound = atof(argv[argv_id++]);
    int max_iter = atoi(argv[argv_id++]);
    gs_plot_gap = atoi(argv[argv_id++]);
    int verb = atoi(argv[argv_id++]);

    double Du = 0.2;
    double Dv = 0.1;
    double F = 0.01;
    double k = 0.05;
    double dt = 2.0;

    using T = double;
    size_t nbEle = L * L * L;
    size_t buffer_size = (L + 2) * (L + 2) * (L + 2);
    GrayScott gs(L, Du, Dv, dt, F, k, errorBound);

    T * u = (T *)malloc(buffer_size * sizeof(T));
    T * v = (T *)malloc(buffer_size * sizeof(T));
    T * u2 = (T *)malloc(buffer_size * sizeof(T));
    T * v2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char * u_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));
    unsigned char * v_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    gs.initData_noghost(u, v, u2, v2);

    size_t u_cmpSize = 0, v_cmpSize = 0;
    SZp_compress_3dLorenzo(u, u_cmpData, L, L, L, blockSideLength, errorBound, u_cmpSize);
    SZp_compress_3dLorenzo(v, v_cmpData, L, L, L, blockSideLength, errorBound, v_cmpSize);

    SZp_grayscott_3dLorenzo<T>(Du, Dv, F, k, dt, u_cmpData, v_cmpData, L, blockSideLength, max_iter, errorBound, u_cmpSize, v_cmpSize, decmpState::full, verb);
    SZp_grayscott_3dLorenzo<T>(Du, Dv, F, k, dt, u_cmpData, v_cmpData, L, blockSideLength, max_iter, errorBound, u_cmpSize, v_cmpSize, decmpState::prePred, verb);
    gs.initData(u, v, u2, v2);
    gs.doWork(u, v, u2, v2, max_iter, gs_plot_gap);

    free(u);
    free(v);
    free(u2);
    free(v2);
    free(u_cmpData);
    free(v_cmpData);

    return 0;
}
