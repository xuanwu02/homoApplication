#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    int dim = atoi(argv[argv_id++]);
    std::string ht_config(argv[argv_id++]);

    switch(dim){
        case 2:{
            heatdis2d_data_dir = argv[argv_id++];
            ht2DSettings s = ht2DSettings::from_json(ht_config);
            ht2d_plot_gap = s.plotgap;
            ht2d_plot_offset = s.offset;
            ht2d_criteria = s.criteria;

            using T = float;
            size_t buffer_size = (s.dim1 + 2) * (s.dim2 + 2);
            HeatDis2D heatdis(s.src_temp, s.wall_temp, s.ratio, s.dim1, s.dim2);

            T * h = (T *)malloc(buffer_size * sizeof(T));
            T * h2 = (T *)malloc(buffer_size * sizeof(T));

            heatdis.initData(h, h2, s.init_temp);
            heatdis.doWork(h, h2, s.criteria, s.steps, s.plotgap, s.offset);

            free(h);
            free(h2);
            break;
        }
        case 3:{
            heatdis3d_data_dir = argv[argv_id++];
            ht3DSettings s = ht3DSettings::from_json(ht_config);
            ht3d_plot_gap = s.plotgap;
            ht3d_plot_offset = s.offset;
            ht3d_criteria = s.criteria;

            using T = float;
            size_t buffer_size = (s.dim1 + 2) * (s.dim2 + 2) * (s.dim3 + 2);
            HeatDis3D heatdis(s.T_top, s.T_bott, s.T_wall, s.alpha, s.dim1, s.dim2, s.dim3);

            T * h = (T *)malloc(buffer_size * sizeof(T));
            T * h2 = (T *)malloc(buffer_size * sizeof(T));

            heatdis.initData(h, h2, s.T_init);
            heatdis.doWork(h, h2, s.criteria, s.steps, s.plotgap, s.offset);

            free(h);
            free(h2);
            break;
        }
        default:
            break;
    }

    return 0;
}
