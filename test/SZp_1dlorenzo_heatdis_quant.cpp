#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "SZp_application_entry.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    int blockSideLength = atoi(argv[argv_id++]);
    if((dim1 % blockSideLength) || (dim2 % blockSideLength)){
        printf("incompatible blockSideLength\n");
        exit(0);
    }
    double errorBound = atof(argv[argv_id++]);
    int max_iter = atoi(argv[argv_id++]);
    size_t nbEle = dim1 * dim2;

    using T = float;
    T * oriData = (T *)calloc(nbEle, sizeof(T));
    initData(dim1, dim2, oriData);
    unsigned char *cmpData = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, errorBound, &cmpSize);
    T * h = (T *)calloc(nbEle, sizeof(T));
    doWork(dim1, dim2, max_iter, oriData, h);

    SZp_heatdis_dec2Quant_1dLorenzo(cmpData, &cmpSize, dim1, dim2, blockSideLength, errorBound, max_iter);
    writefile("decdata.dec2quant.dat", cmpData, cmpSize);
    free(cmpData);
    // read compressed file
    size_t num;
    std::vector<unsigned char> compressed = readfile<unsigned char>("decdata.dec2quant.dat", num);
    T * decData = (T *)malloc(nbEle * sizeof(T));
    SZp_decompress_1dLorenzo(decData, compressed.data(), dim1, dim2, blockSideLength, errorBound);
    double max_err = verify(oriData, decData, dim1, dim2);
    printf("cr = %f, max_err = %.14f\n", 1.0 * sizeof(T) * nbEle / cmpSize, max_err);

    free(h);
    free(decData);
    free(cmpData);

    return 0;
}
