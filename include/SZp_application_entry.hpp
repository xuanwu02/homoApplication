#ifndef _SZP_HEATDIS_ENTRY_HPP
#define _SZP_HEATDIS_ENTRY_HPP

#include <iostream>
#include "SZp_lorenzoPredictor1D.hpp"

void SZp_compress_1dLorenzo(
    float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t *cmpSize
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZp_compress_kernel_1dLorenzo(oriData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound, cmpSize);
    free(absQuantDiff);
    free(signFlag);
}

void SZp_decompress_1dLorenzo(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZp_decompress_kernel_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound);
    free(absQuantDiff);
    free(signFlag);
}

/**
 *  Global mean encapsulated
*/
void SZp_decompress_1dLorenzo(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, double& mean
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZp_decompress_kernel_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound, mean);
    free(absQuantDiff);
    free(signFlag);
}

void SZp_heatdis_dec2Quant_1dLorenzo(
    unsigned char *compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, int max_iter
){
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)malloc(block_dim1 * sizeof(int));
        fixedRate[i] = (int *)malloc(block_num * sizeof(int));
        cmpData[i] = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_quant_1dLorenzo(cmpData, offsets, fixedRate, absQuantDiff, signFlag, dim1, dim2, blockSideLength, errorBound, cmpSize, max_iter);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f\n", elapsed_time);
    int status = max_iter % 2;
    memcpy(compressed_data, cmpData[status], 4 * nbEle * sizeof(unsigned char));
    for(int i=0; i<2; i++){
        free(offsets[i]);
        free(fixedRate[i]);
        free(cmpData[i]);
    }
    free(offsets);
    free(fixedRate);
    free(cmpData);
    free(absQuantDiff);
    free(signFlag);
}

void SZp_heatdis_dec2Lorenzo_1dLorenzo(
    unsigned char *compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, int max_iter
){
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)malloc(block_dim1 * sizeof(int));
        fixedRate[i] = (int *)malloc(block_num * sizeof(int));
        cmpData[i] = (unsigned char *)malloc(4 * nbEle * sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_lorenzo_1dLorenzo(cmpData, offsets, fixedRate, absQuantDiff, signFlag, dim1, dim2, blockSideLength, errorBound, cmpSize, max_iter);
    clock_gettime(CLOCK_REALTIME, &end);
    double elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("elapsed_time = %.6f\n", elapsed_time);
    int status = max_iter % 2;
    memcpy(compressed_data, cmpData[status], 4 * nbEle * sizeof(unsigned char));
    for(int i=0; i<2; i++){
        free(offsets[i]);
        free(fixedRate[i]);
        free(cmpData[i]);
    }
    free(offsets);
    free(fixedRate);
    free(cmpData);
    free(absQuantDiff);
    free(signFlag);
}

double SZp_mean_dec2Quant_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    int *quantInds = (int *)malloc(blockSideLength * dim2 * sizeof(int));
    int quant_sum = SZp_mean_kernel_quant_1dLorenzo(cmpData, dim1, dim2, absQuantDiff, signFlag, quantInds, blockSideLength, errorBound);
    free(absQuantDiff);
    free(signFlag);
    free(quantInds);
    return 2 * (quant_sum / nbEle) * errorBound;
}

double SZp_mean_dec2Lorenzo_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    int *lorenzoPred = (int *)malloc(blockSideLength * dim2 * sizeof(int));
    int quant_sum = SZp_mean_kernel_lorenzo_1dLorenzo(cmpData, dim1, dim2, absQuantDiff, signFlag, lorenzoPred, blockSideLength, errorBound);
    free(absQuantDiff);
    free(signFlag);
    free(lorenzoPred);
    return 2 * (quant_sum / nbEle) * errorBound;
}

void SZp_derivative_dec2Quant_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    double *dx_result, double *dy_result
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZp_derivative_quant_1dLorenzo(cmpData, dim1, dim2, blockSideLength, errorBound, absQuantDiff, signFlag, dx_result, dy_result);
    free(absQuantDiff);
    free(signFlag);
}

void SZp_derivative_dec2Lorenzo_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    double *dx_result, double *dy_result
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = nbEle / blockSize;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZp_derivative_lorenzo_1dLorenzo(cmpData, dim1, dim2, blockSideLength, errorBound, absQuantDiff, signFlag, dx_result, dy_result);
    free(absQuantDiff);
    free(signFlag);
}

#endif