#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_2D.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    std::string data_file(argv[argv_id++]);
    size_t dim = atoi(argv[argv_id++]);
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    size_t dim3 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    double eb = atof(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);
    if(dim == 2) dim3 = 1;

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));

    size_t cmpSize = 0;
    SZp_compress(oriData, cmpData, dim1, dim2, blockSideLength, eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    double var = SZp_variance(cmpData, dim1, dim2, decData, blockSideLength, eb, intToDecmpState(stateType));
    printf("variance = %.6f\n", var);

    // double act_mean = 0;
    // for(size_t i=0; i<nbEle; i++) act_mean += oriData[i];
    // act_mean /= nbEle;
    // double act_var = 0;
    // for(size_t i=0; i<nbEle; i++) act_var += (oriData[i] - act_mean) * (oriData[i] - act_mean);
    // act_var /= (nbEle - 1);
    // printf("error = %.6f\n", fabs(act_var - var));

    SZp_decompress(decData, cmpData, dim1, dim2, blockSideLength, eb);
    double doc_mean = 0;
    for(size_t i=0; i<nbEle; i++) doc_mean += decData[i];
    doc_mean /= nbEle;
    double doc_var = 0;
    for(size_t i=0; i<nbEle; i++) doc_var += (decData[i] - doc_mean) * (decData[i] - doc_mean);
    doc_var /= (nbEle - 1);
    printf("rel error = %.6e\n", fabs((doc_var - var) / eb));

    free(decData);
    free(cmpData);

    return 0;
}
