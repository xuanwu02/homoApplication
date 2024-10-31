#ifndef _SZP_HEATDIS_ENTRY_HPP
#define _SZP_HEATDIS_ENTRY_HPP

#include <iostream>
#include "SZp_heatdis_rowwise_1b.hpp"
#include "SZp_heatdis_rowwise_2b.hpp"

void SZp_compress_rowwise_1d_block(
    float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound, size_t *cmpSize
){
    size_t nbEle = dim1 * dim2;
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int block_num = row_block_num * dim1;
    unsigned int *absLorenzo = (unsigned int *)malloc(sizeof(unsigned int) * nbEle);
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    SZp_compress_kernel_rowwise_1d_block(oriData, cmpData, dim1, dim2, absLorenzo, signFlag, errorBound, blockSize, cmpSize);
    free(absLorenzo);
    free(signFlag);
}

void SZp_decompress_rowwise_1d_block(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound
){
    size_t nbEle = dim1 * dim2;
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int block_num = row_block_num * dim1;
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    SZp_decompress_kernel_rowwise_1d_block(decData, cmpData, dim1, dim2, absLorenzo, signFlag, errorBound, blockSize);
    free(absLorenzo);
    free(signFlag);
}

void SZp_heatdis_dec2Quant_rowwise_1d_block(
    unsigned char *compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound, int max_iter
){
    int block_num = dim1;
    size_t nbEle = dim1 * dim2;
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)calloc(block_num, sizeof(int));
        fixedRate[i] = (int *)calloc(block_num, sizeof(int));
        cmpData[i] = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));
    unsigned int *absLorenzo = (unsigned int *)calloc(blockSize, sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)calloc(blockSize, sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_quant_rowwise_1d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, blockSize, errorBound, cmpSize, max_iter);
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
    free(absLorenzo);
    free(signFlag);
}

void SZp_heatdis_dec2Lorenzo_rowwise_1d_block(
    unsigned char * compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound, int max_iter
){
    int block_num = dim1;
    size_t nbEle = dim1 * dim2;
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)calloc(block_num, sizeof(int));
        fixedRate[i] = (int *)calloc(block_num, sizeof(int));
        cmpData[i] = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));
    unsigned int *absLorenzo = (unsigned int *)calloc(blockSize, sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)calloc(blockSize, sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_lorenzo_rowwise_1d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, blockSize, errorBound, cmpSize, max_iter);
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
    free(absLorenzo);
    free(signFlag);
}

void SZp_compress_rowwise_2d_block(
    float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t *cmpSize
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    SZp_compress_kernel_rowwise_2d_block(oriData, cmpData, dim1, dim2, blockSideLength, absLorenzo, signFlag, errorBound, cmpSize);
    free(absLorenzo);
    free(signFlag);
}

void SZp_decompress_rowwise_2d_block(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    SZp_decompress_kernel_rowwise_2d_block(decData, cmpData, dim1, dim2, blockSideLength, absLorenzo, signFlag, errorBound);
    free(absLorenzo);
    free(signFlag);
}

void SZp_heatdis_dec2Quant_rowwise_2d_block(
    unsigned char *compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, int max_iter
){
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_num = nbEle / (blockSideLength * blockSideLength);
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)calloc(block_dim1, sizeof(int));
        fixedRate[i] = (int *)calloc(block_num, sizeof(int));
        cmpData[i] = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_quant_rowwise_2d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, blockSideLength, errorBound, cmpSize, max_iter);
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
    free(absLorenzo);
    free(signFlag);
}

void SZp_heatdis_dec2Lorenzo_rowwise_2d_block(
    unsigned char *compressed_data, size_t *cmpSize,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, int max_iter)
{
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_num = nbEle / (blockSideLength * blockSideLength);
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2 * sizeof(int *));
    int **fixedRate = (int **)malloc(2 * sizeof(int *));
    for(int i=0; i<2; i++){
        offsets[i] = (int *)calloc(block_dim1, sizeof(int));
        fixedRate[i] = (int *)calloc(block_num, sizeof(int));
        cmpData[i] = (unsigned char *)calloc(4 * nbEle, sizeof(unsigned char));
    }
    memcpy(cmpData[0], compressed_data, 4 * nbEle * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_kernel_lorenzo_rowwise_2d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, blockSideLength, errorBound, cmpSize, max_iter);
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
    free(absLorenzo);
    free(signFlag);
}

#endif