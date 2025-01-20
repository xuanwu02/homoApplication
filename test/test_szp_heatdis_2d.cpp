#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor2D.hpp"
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    std::string ht_config(argv[1]);
    htSettings s = htSettings::from_json(ht_config);

    using T = float;
    size_t nbEle = s.dim1 * s.dim1;
    size_t buffer_size = (s.dim1 + 2) * (s.dim1 + 2);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    HeatDis heatdis(s.src_temp, s.wall_temp, s.ratio, s.dim1, s.dim1);
    heatdis.initData_noghost(h, h2, s.init_temp);
    size_t cmpSize = 0;
    SZp_compress_2dLorenzo(h, cmpData, s.dim1, s.dim1, s.B, s.eb, cmpSize);
    SZp_heatdis_2dLorenzo<T>(cmpData, s.dim1, s.dim1, s.B, s.steps, cmpSize, s.src_temp, s.wall_temp, s.init_temp, s.ratio, s.eb, intToDecmpState(s.stateType), false);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
