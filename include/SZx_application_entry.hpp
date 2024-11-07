#ifndef _SZX_MEAN_BASED_ENTRY_HPP
#define _SZX_MEAN_BASED_ENTRY_HPP

#include <iostream>
#include "SZx_MeanPredictor.hpp"

template <class T>
void SZx_compress_2dblock(
    T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t *cmpSize
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZx_compress_kernel_2dblock(oriData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound, cmpSize);
    free(absQuantDiff);
    free(signFlag);
}

template <class T>
void SZx_decompress_2dblock(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZx_decompress_kernel_2dblock(decData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound);
    free(absQuantDiff);
    free(signFlag);
}

/**
 * Global mean encapsulated
*/
template <class T>
void SZx_decompress_2dblock(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, double& mean
){
    int blockSize = blockSideLength * blockSideLength;
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));
    SZx_decompress_kernel_2dblock(decData, cmpData, dim1, dim2, blockSideLength, absQuantDiff, signFlag, errorBound, mean);
    free(absQuantDiff);
    free(signFlag);
}

double SZx_mean_2dblock(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    unsigned char * qmean_pos = cmpData + block_num;
    // size_t global_mean = 0;
    long int global_mean = 0;
    for(int k=0; k<block_num; k++){
        int quant_mean = (0xff000000 & (*qmean_pos << 24)) |
                        (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                        (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                        (0x000000ff & *(qmean_pos+3));
        global_mean += quant_mean;
        qmean_pos += 4;
    }
    return 2 * (1.0 * global_mean / block_num) * errorBound;
}

void SZx_derivative_quant_2dblock(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double *dx_result, double *dy_result,
    double errorBound, double& elapsed_time
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int * fixedRate = (int *)malloc(block_num * sizeof(int));
    int * offsets = (int *)malloc((block_dim1 + 1) * sizeof(int));
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_derivative_quant_kernel_2dblock(cmpData, fixedRate, offsets, absQuantDiff, signFlag, dim1, dim2, blockSideLength, dx_result, dy_result, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;

    free(offsets);
    free(fixedRate);
    free(absQuantDiff);
    free(signFlag);
}

void SZx_derivative_diff_2dblock(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double *dx_result, double *dy_result,
    double errorBound, double& elapsed_time
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int * fixedRate = (int *)malloc(block_num * sizeof(int));
    int * offsets = (int *)malloc((block_dim1 + 1) * sizeof(int));
    unsigned int *absQuantDiff = (unsigned int *)malloc(blockSize * sizeof(unsigned int));
    unsigned char *signFlag = (unsigned char *)malloc(blockSize * sizeof(unsigned char));

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    SZx_derivative_diff_kernel_2dblock(cmpData, fixedRate, offsets, absQuantDiff, signFlag, dim1, dim2, blockSideLength, dx_result, dy_result, errorBound);
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;

    free(offsets);
    free(fixedRate);
    free(absQuantDiff);
    free(signFlag);
}

#endif