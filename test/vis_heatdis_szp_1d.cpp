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
    heatdis2d_data_dir = argv[argv_id++];
    ht2DSettings s = ht2DSettings::from_json(ht_config);
    ht2d_plot_gap = s.plotgap;
    ht2d_plot_offset = s.offset;
    ht2d_criteria = s.criteria;

    printf("2D heat distribution (1D lorenzo): B = %d, eb = %g, gap = %d\n", s.B, s.eb, s.plotgap);

    using T = float;
    size_t nbEle = s.dim1 * s.dim2;
    size_t buffer_size = (s.dim1 + 2) * (s.dim2 + 2);
    HeatDis2D heatdis(s.src_temp, s.wall_temp, s.ratio, s.dim1, s.dim2);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    heatdis.initData_noghost(h, h2, s.init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, cmpData, s.dim1, s.dim2, s.B, s.eb, cmpSize);

    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::prePred, true);
    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::postPred, true);
    SZp_heatdis_1dLorenzo<T>(cmpData, s, decmpState::full, true);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
