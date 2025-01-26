#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor3D.hpp"
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    std::string gs_config(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    gsSettings s = gsSettings::from_json(gs_config);

    using T = double;
    size_t nbEle = s.L * s.L * s.L;
    size_t buffer_size = (s.L + 2) * (s.L + 2) * (s.L + 2);
    GrayScott gs(s.L, s.Du, s.Dv, s.dt, s.F, s.k, s.eb);

    T * u = (T *)malloc(buffer_size * sizeof(T));
    T * v = (T *)malloc(buffer_size * sizeof(T));
    T * u2 = (T *)malloc(buffer_size * sizeof(T));
    T * v2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char * u_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));
    unsigned char * v_cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    gs.initData_noghost(u, v, u2, v2);

    size_t u_cmpSize = 0, v_cmpSize = 0;
    SZp_compress_3dLorenzo(u, u_cmpData, s.L, s.L, s.L, s.B, s.eb, u_cmpSize);
    SZp_compress_3dLorenzo(v, v_cmpData, s.L, s.L, s.L, s.B, s.eb, v_cmpSize);

    SZp_grayscott_3dLorenzo<T>(u_cmpData, v_cmpData, s, intToDecmpState(stateType), true);

    free(u);
    free(v);
    free(u2);
    free(v2);
    free(u_cmpData);
    free(v_cmpData);

    return 0;
}
