#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor1D.hpp"
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    std::string ht_config(argv[1]);
    htSettings s = htSettings::from_json(ht_config);

    using T = float;
    size_t nbEle = s.dim1 * s.dim2;
    size_t buffer_size = (s.dim1 + 2) * (s.dim2 + 2);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    HeatDis heatdis(s.src_temp, s.wall_temp, s.ratio, s.dim1, s.dim2);
    heatdis.initData_noghost(h, h2, s.init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, cmpData, s.dim1, s.dim2, s.B, s.eb, cmpSize);

    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::full, true);
    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::prePred, true);
    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::postPred, true);
    heatdis.initData(h, h2, s.init_temp);
    heatdis.doWork(h, h2, s.criteria, s.steps, s.plotgap, s.offset);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
