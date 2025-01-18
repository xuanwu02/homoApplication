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
    int type = atoi(argv[argv_id++]);
    decmpState state = intToDecmpState(type);
    int max_iter = atoi(argv[argv_id++]);    

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
    T * u_decData = (T *)malloc(buffer_size * sizeof(T));
    T * v_decData = (T *)malloc(buffer_size * sizeof(T));
    unsigned char * u_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));
    unsigned char * v_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    gs.initData_noghost(u, v, u2, v2);

    size_t u_cmpSize = 0, v_cmpSize = 0;
    SZp_compress_3dLorenzo(u, u_cmpData, L, L, L, blockSideLength, errorBound, u_cmpSize);
    SZp_compress_3dLorenzo(v, v_cmpData, L, L, L, blockSideLength, errorBound, v_cmpSize);
    SZp_grayscott_3dLorenzo<T>(Du, Dv, F, k, dt, u_cmpData, v_cmpData, L, blockSideLength, max_iter, errorBound, u_cmpSize, v_cmpSize, state);

    auto uCmpVec = readfile<unsigned char>("u.cmp.dat", u_cmpSize);
    auto vCmpVec = readfile<unsigned char>("v.cmp.dat", v_cmpSize);

    switch(state){
        case decmpState::full:{
            SZp_decompress_3dLorenzo(u, uCmpVec.data(), L+2, L+2, L+2, blockSideLength, errorBound);
            SZp_decompress_3dLorenzo(v, vCmpVec.data(), L+2, L+2, L+2, blockSideLength, errorBound);
            gs.trimData(u, u_decData);
            gs.trimData(v, v_decData);
            break;
        }
        case decmpState::prePred:{
            SZp_decompress_3dLorenzo(u_decData, uCmpVec.data(), L, L, L, blockSideLength, errorBound);
            SZp_decompress_3dLorenzo(v_decData, vCmpVec.data(), L, L, L, blockSideLength, errorBound);
            break;
        }
        case decmpState::postPred:{
            break;
        }
    }

    gs.initData(u, v, u2, v2);
    gs.doWork(u, v, u2, v2, max_iter);
    gs.trimData(u, u2);
    gs.trimData(v, v2);
    double u_err = verify(u2, u_decData, L, L, L);
    printf("U max_error = %.6f\n", u_err);
    double v_err = verify(v2, v_decData, L, L, L);
    printf("V max_error = %.6f\n", v_err);

    writefile("u.dec", u2, nbEle);
    writefile("v.dec", v2, nbEle);
    writefile("u1.dec", u_decData, nbEle);
    writefile("v1.dec", v_decData, nbEle);

    free(u);
    free(v);
    free(u2);
    free(v2);
    free(u_decData);
    free(v_decData);
    free(u_cmpData);
    free(v_cmpData);

    return 0;
}
