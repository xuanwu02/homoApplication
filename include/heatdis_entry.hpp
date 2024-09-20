#ifndef _STENCIL_HEATDIS_ENTRY_HPP
#define _STENCIL_HEATDIS_ENTRY_HPP

#include <iostream>
#include "heatdis_1b.hpp"

void SZp_compress_1D(float *oriData, unsigned char *cmpData,
                     size_t dim1, size_t dim2,
                     double errorBound, int blockSize, size_t *cmpSize)
{
    size_t nbEle = dim1 * dim2;
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int block_num = row_block_num * dim1;
    unsigned int *absLorenzo = (unsigned int *)malloc(sizeof(unsigned int) * nbEle);
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    int *fixedRate = (int *)malloc(sizeof(int) * block_num);
    SZp_compress_kernel_1D(oriData, cmpData, dim1, dim2, absLorenzo, signFlag, fixedRate, errorBound, blockSize, cmpSize);
    free(absLorenzo);
    free(signFlag);
    free(fixedRate);
}

void SZp_decompress_1D(float *decData, unsigned char *cmpData,
                       size_t dim1, size_t dim2,
                       double errorBound, int blockSize)
{
    size_t nbEle = dim1 * dim2;
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int block_num = row_block_num * dim1;
    unsigned int *absLorenzo = (unsigned int *)calloc(nbEle, sizeof(int));
    unsigned char *signFlag = (unsigned char *)calloc(nbEle, sizeof(unsigned char));
    int *fixedRate = (int *)malloc(sizeof(int) * block_num);
    SZp_decompress_kernel_1D(decData, cmpData, dim1, dim2, absLorenzo, signFlag, fixedRate, errorBound, blockSize);
    free(absLorenzo);
    free(signFlag);
    free(fixedRate);
}

void SZp_heatdis_decompressToQuant(unsigned char * compressed_data, size_t dim1, size_t dim2, double errorBound, int blockSize, size_t *cmpSize, int max_iter)
{
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
    SZp_heatdis_kernel_decomressToQuant(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, errorBound, blockSize, cmpSize, max_iter);
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

void SZp_heatdis_decompressToLorenzo(unsigned char * compressed_data, size_t dim1, size_t dim2, double errorBound, int blockSize, size_t *cmpSize, int max_iter)
{
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
    SZp_heatdis_kernel_decomressToLorenzo(cmpData, offsets, fixedRate, absLorenzo, signFlag, dim1, dim2, errorBound, blockSize, cmpSize, max_iter);
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