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
    std::string ht_config(argv[argv_id++]);
    heatdis3d_data_dir = argv[argv_id++];
    ht3DSettings s = ht3DSettings::from_json(ht_config);
    ht3d_plot_gap = s.plotgap;
    ht3d_plot_offset = s.offset;
    ht3d_criteria = s.criteria;

    printf("3D heat distribution (3D lorenzo): B = %d, eb = %g, gap = %d\n", s.B, s.eb, s.plotgap);

    using T = float;
    size_t buffer_size = (s.dim1 + 2) * (s.dim2 + 2) * (s.dim3 + 2);
    HeatDis3D heatdis(s.T_top, s.T_bott, s.T_wall, s.alpha, s.dim1, s.dim2, s.dim3);

    T * h = (T *)malloc(buffer_size * sizeof(T));
    T * h2 = (T *)malloc(buffer_size * sizeof(T));
    unsigned char *cmpData = (unsigned char *)malloc(buffer_size * sizeof(T));

    heatdis.initData_noghost(h, h2, s.T_init);
    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(h, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, cmpSize);

    SZp_heatdis_3dLorenzo<T>(cmpData, s, decmpState::prePred, true);
    SZp_heatdis_3dLorenzo<T>(cmpData, s, decmpState::full, true);

    free(h);
    free(h2);
    free(cmpData);

    return 0;
}
