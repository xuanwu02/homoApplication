#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZx_1D.hpp"
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
    std::string rel_eb(argv[argv_id]);
    double eb = atof(argv[argv_id++]);
    int stateType = atoi(argv[argv_id++]);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == dim1 * dim2 * dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, eb);

    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));
    int * intData = (int *)malloc(nbEle * sizeof(int));

    struct timespec start, end;
    double elapsed_time;

    size_t cmpSize = 0;
    SZx_compress(oriData, cmpData, dim1, dim2, dim3, blockSideLength, eb, cmpSize);

    clock_gettime(CLOCK_REALTIME, &start);
    switch(stateType){
        case 0:{
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
            break;
        }
        case 1:{
            SZx_decompress_prePred(intData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
            break;
        }
        case 2:{
            SZx_decompress_postPred(intData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
            break;
        }
        case 3:{
            SZx_decompress_meta(intData, cmpData, dim1, dim2, dim3, blockSideLength, eb);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("eb=%s, state=%d, elapsed_time=%.6f\n", rel_eb.c_str(), stateType, elapsed_time);

    free(decData);
    free(intData);
    free(cmpData);

    return 0;
}
