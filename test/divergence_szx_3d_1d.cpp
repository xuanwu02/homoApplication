#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZx_1D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    std::string data_dir(argv[argv_id++]);
    size_t dim = atoi(argv[argv_id++]);
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    size_t dim3 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double eb = atof(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);

    using T = float;
    int i;

    auto data_files = getFiles(data_dir);
    assert(data_files.size() == 3);
    size_t nbEle;
    std::vector<std::vector<T>> oriData_vec(3);
    std::vector<T *> oriData(3, nullptr);
    for(i=0; i<dim; i++){
        oriData_vec[i].resize(dim1 * dim2 * dim3);
        oriData_vec[i] = readfile<T>(data_files[i].c_str(), nbEle);
        assert(nbEle == dim1 * dim2 * dim3);
        oriData[i] = oriData_vec[i].data();
    }
    set_overall_eb(oriData[0], oriData[1], oriData[2], nbEle, eb);

    std::array<unsigned char *, 3> cmpData = {nullptr, nullptr, nullptr};
    std::vector<T *> decData(dim, nullptr);
    for(i=0; i<dim; i++){
        cmpData[i] = (unsigned char *)malloc(nbEle * sizeof(T));
        decData[i] = (T *)malloc(nbEle * sizeof(T));
    }
    T * divergence_result = (T *)malloc(nbEle * sizeof(T));
    T * ref_divergence_result = (T *)malloc(nbEle * sizeof(T));


    size_t cmpSize = 0, compressed_size = 0;
    for(i=0; i<dim; i++){
        SZx_compress(oriData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, eb, compressed_size);
        cmpSize += compressed_size;
    }
    printf("cr = %.2f\n", 1.0 * dim * nbEle * sizeof(T) / cmpSize);
    for(i=0; i<dim; i++){
        std::vector<T>().swap(oriData_vec[i]);
    }

    SZx_divergence3D(cmpData, dim1, dim2, dim3, blockSideLength, eb, divergence_result, intToDecmpState(stateType));
    for(i=0; i<dim; i++){
        SZx_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, eb);
    }
    compute_divergence_3d(dim1, dim2, dim3, decData[0], decData[1], decData[2], ref_divergence_result);
    double err = verify_dxdydz(ref_divergence_result, divergence_result, dim1, dim2, dim3);
    printf("max error = (%.6e, 0, 0)\n", err/eb);

    for(i=0; i<dim; i++){
        free(decData[i]);
        free(cmpData[i]);
    }
    free(divergence_result);
    free(ref_divergence_result);

    return 0;
}
