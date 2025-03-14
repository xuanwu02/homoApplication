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
    int argv_id = 1;
    std::string ht_config(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    ht2DSettings s = ht2DSettings::from_json(ht_config);
    ht2d_plot_gap = s.plotgap;
    ht2d_plot_offset = s.offset;
    ht2d_criteria = s.criteria;

    printf("2D heat distribution (1D lorenzo): stateType = %d, B = %d, eb = %g, gap = %d\n", stateType, s.B, s.eb, s.plotgap);

    using T = float;
    size_t nbEle = s.dim1 * s.dim1;
    size_t buffer_size = (s.dim1 + 2) * (s.dim1 + 2);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    HeatDis2D heatdis(s.src_temp, s.wall_temp, s.ratio, s.dim1, s.dim1);
    heatdis.initData_noghost(h, h2, s.init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, cmpData, s.dim1, s.dim1, s.B, s.eb, cmpSize);
    SZp_heatdis_1dLorenzo<T>(cmpData, s, intToDecmpState(stateType), false);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
