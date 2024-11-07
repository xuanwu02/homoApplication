#ifndef _SZX_MEAN_BASED_2D_HPP
#define _SZX_MEAN_BASED_2D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "application_utils.hpp"

template <class T>
inline int SZp_quantize(const T& data, double errorBound)
{
    return static_cast<int>(std::floor((data + errorBound) / (2 * errorBound)));
}

template <class T>
void SZx_compress_kernel_2dblock(
    T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    double errorBound, size_t *cmpSize
){
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    unsigned char * qmean_pos = cmpData + block_num;
    unsigned char * encode_pos = cmpData + 5 * block_num;
    std::vector<int> block_quant_inds(blockSize, 0);
    int x, y, i, j;
    for(x=0; x<block_dim1; x++){
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate, quant_mean = 0;
            int max_quant_diff = 0;
            for(i=0; i<blockSideLength; i++){
                for(j=0; j<blockSideLength; j++){
                    int global_index = (x * blockSideLength + i) * dim2 + y * blockSideLength + j;
                    int curr_quant = SZp_quantize(oriData[global_index], errorBound);
                    int local_index = i * blockSideLength + j;
                    block_quant_inds[local_index] = curr_quant;
                    quant_mean += curr_quant;
                }
            }
            quant_mean /= blockSize;
            for(i=0; i<blockSideLength; i++){
                for(j=0; j<blockSideLength; j++){
                    int local_index = i * blockSideLength + j;
                    int quant_diff = block_quant_inds[local_index] - quant_mean;
                    signFlag[local_index] = (quant_diff < 0);
                    absQuantDiff[local_index] = abs(quant_diff);
                    max_quant_diff = max_quant_diff > absQuantDiff[local_index] ? max_quant_diff : absQuantDiff[local_index];
                }
            }
            temp_fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
            cmpData[block_index] = (unsigned char)temp_fixed_rate;
            for(int k=3; k>=0; k--){
                *(qmean_pos++) = (quant_mean >> (8 * k)) & 0xff;
            }
            if(temp_fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, blockSize, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absQuantDiff, blockSize, encode_pos, temp_fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
        }
    }
    *cmpSize = encode_pos - cmpData;
}

template <class T>
void SZx_decompress_kernel_2dblock(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    double errorBound
){
    size_t nbEle = dim1 * dim2;
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    unsigned char * qmean_pos = cmpData + block_num;
    unsigned char * encode_pos = cmpData + 5 * block_num;
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    std::vector<int> blocks_quant_mean(block_num, 0);
    for(int k=0; k<block_num; k++){
        blocks_quant_mean[k] = (0xff000000 & (*qmean_pos << 24)) |
                                (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                                (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                                (0x000000ff & *(qmean_pos+3));
        qmean_pos += 4;
    }
    int x, y, i, j;
    for(x=0; x<block_dim1; x++){
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate = (int)cmpData[block_index];
            int quant_mean = blocks_quant_mean[block_index];
            if(temp_fixed_rate){
                convertByteArray2IntArray_fast_1b_args(blockSize, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, blockSize, absQuantDiff, temp_fixed_rate);
                encode_pos += savedbitsbytelength;
                for(i=0; i<blockSideLength; i++){
                    for(j=0; j<blockSideLength; j++){
                        int local_index = i * blockSideLength + j;
                        int sign = -(int)signFlag[local_index];
                        int quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                        int curr_quant = quant_diff + quant_mean;
                        int global_index = (x * blockSideLength + i) * dim2 + y * blockSideLength + j;
                        decData[global_index] = 2 * curr_quant * errorBound;
                    }
                }
            }
            else{
                for(i=0; i<blockSideLength; i++){
                    for(j=0; j<blockSideLength; j++){
                        int global_index = (x * blockSideLength + i) * dim2 + y * blockSideLength + j;
                        decData[global_index] = 2 * quant_mean * errorBound;
                    }
                }
            }
        }
    }
}

/**
 * blockwise-serial
 */
void SZx_recoverToDiff_blockRow(
    int blockRow_ind, int block_dim2,
    int blockSideLength, int blockSize, int *fixedRate,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    unsigned char *cmpData_pos, int *quantDiff
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    int y, i, j;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int quant_diff;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            for(i=0; i<blockSideLength; i++){
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    local_index = local_offset + j;
                    global_index = global_offset + local_index;
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    quantDiff[global_index] = quant_diff;
                }
            }
        }
        else{
            for(i=0; i<blockSideLength; i++){
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    global_index = global_offset + local_offset + j;
                    quantDiff[global_index] = 0;
                }
            }
        }
    }
}

template <class T>
inline void derivative_process_diff_nw_corner_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *rightBlock, int *bottomBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int left, right, top, bottom;
    int curr_mean = blocks_mean[block_index], right_mean = blocks_mean[block_index + 1], bottom_mean = blocks_mean[block_index + block_dim2];
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
inline void derivative_process_diff_ne_corner_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *bottomBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], left_mean = blocks_mean[block_index - 1], bottom_mean = blocks_mean[block_index + block_dim2];
    int row_offset = (block_dim2 - 1) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
inline void derivative_process_diff_sw_corner_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *rightBlock, int *topBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], right_mean = blocks_mean[block_index + 1], top_mean = blocks_mean[block_index - block_dim2];
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[index - blockSideLength + 1] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[index - blockSideLength + 1] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[index - blockSideLength + 1] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
    }
}

template <class T>
inline void derivative_process_diff_se_corner_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *topBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], left_mean = blocks_mean[block_index - 1], top_mean = blocks_mean[block_index - block_dim2];
    int row_offset = (block_dim2 - 1) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
    }
}

template <class T>
inline void derivative_process_diff_topRow_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *rightBlock, int *bottomBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], bottom_mean = blocks_mean[block_index + block_dim2];
    int left_mean = blocks_mean[block_index - 1], right_mean = blocks_mean[block_index + 1];
    int row_offset = (block_index % block_dim2) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index]) * errorBound * 2;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
inline void derivative_process_diff_bottomRow_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *rightBlock, int *topBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], top_mean = blocks_mean[block_index - block_dim2];
    int left_mean = blocks_mean[block_index - 1], right_mean = blocks_mean[block_index + 1];
    int row_offset = (block_index % block_dim2) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index] - currBlock[index - blockSideLength]) * errorBound * 2;
        }
    }
}

template <class T>
inline void derivative_process_diff_leftCol_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *topBlock, int *bottomBlock, int *rightBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int left, right, top, bottom;
    int curr_mean = blocks_mean[block_index], right_mean = blocks_mean[block_index + 1];
    int top_mean = blocks_mean[block_index - block_dim2], bottom_mean = blocks_mean[block_index + block_dim2];
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index]) * errorBound * 2;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
inline void derivative_process_diff_rightCol_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *topBlock, int *bottomBlock, int *leftBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index], left_mean = blocks_mean[block_index - 1];
    int top_mean = blocks_mean[block_index - block_dim2], bottom_mean = blocks_mean[block_index + block_dim2];
    int row_offset = (block_dim2 - 1) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index] - currBlock[index - 1]) * errorBound * 2;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
inline void derivative_process_diff_inner_block(
    size_t dim2, int block_index, int block_dim2,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *rightBlock,
    int *topBlock, int *bottomBlock,
    int *blocks_mean, double errorBound,
    T *dx_pos, T *dy_pos
){
    int i, j;
    int index, res_index;
    int curr_mean = blocks_mean[block_index];
    int left_mean = blocks_mean[block_index - 1], right_mean = blocks_mean[block_index + 1];
    int top_mean = blocks_mean[block_index - block_dim2], bottom_mean = blocks_mean[block_index + block_dim2];
    int row_offset = (block_index % block_dim2) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = ((currBlock[index + blockSideLength] - topBlock[index + blockSideLength * (blockSideLength - 1)]) + (curr_mean - top_mean)) * errorBound;
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (currBlock[index + blockSideLength] - currBlock[index - blockSideLength]) * errorBound;
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = ((currBlock[index + 1] - leftBlock[index + blockSideLength - 1]) + (curr_mean - left_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j, res_index = i * dim2 + row_offset + j;
            dx_pos[res_index] = (currBlock[index + 1] - currBlock[index - 1]) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j, res_index = i * dim2 +row_offset + j;
            dx_pos[res_index] = (rightBlock[i * blockSideLength] - currBlock[index - 1] + (right_mean - curr_mean)) * errorBound;
            dy_pos[res_index] = (bottomBlock[j] - currBlock[index - blockSideLength] + (bottom_mean - curr_mean)) * errorBound;
        }
    }
}

template <class T>
void SZx_compute_derivative_diff_2dblock(
    unsigned char *cmpData, int block_dim1, int block_dim2,
    int blockSideLength, int *blocks_quant_mean, int *offsets, 
    int *fixedRate, unsigned int *absQuantDiff, unsigned char *signFlag,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    T *dx_result, T *dy_result, double errorBound
){
    size_t dim2 = block_dim2 * blockSideLength;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = block_dim1 * block_dim2;
    unsigned char * encode_pos = cmpData + 5 * block_num;
    int * currBlock = nullptr, * tempBlockRow = nullptr;
    int * leftBlock = nullptr, * rightBlock = nullptr, * topBlock = nullptr, * bottomBlock = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    int x, y, block_index;
    {
        x = 0;
        SZx_recoverToDiff_blockRow(x, block_dim2, blockSideLength, blockSize, fixedRate, absQuantDiff, signFlag, encode_pos+offsets[x], currBlockRow);
        SZx_recoverToDiff_blockRow(x+1, block_dim2, blockSideLength, blockSize, fixedRate, absQuantDiff, signFlag, encode_pos+offsets[x+1], nextBlockRow);
        dx_pos = dx_result + x * blockSideLength * dim2, dy_pos = dy_result + x * blockSideLength * dim2;
        {
            y = 0;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_nw_corner_block(dim2, block_index, block_dim2, blockSideLength, currBlock, rightBlock, bottomBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);  
        }
        for(y=1; y<block_dim2-1; y++){
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_topRow_block(dim2, block_index, block_dim2, blockSideLength, currBlock, leftBlock, rightBlock, bottomBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);
        }
        {
            y = block_dim2 - 1;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_ne_corner_block(dim2, block_index, block_dim2, blockSideLength, currBlock, leftBlock, bottomBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);
        }
    }
    for(x=1; x<block_dim1-1; x++){
        tempBlockRow = prevBlockRow;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        nextBlockRow = tempBlockRow;
        SZx_recoverToDiff_blockRow(x+1, block_dim2, blockSideLength, blockSize, fixedRate, absQuantDiff, signFlag, encode_pos+offsets[x+1], nextBlockRow);
        dx_pos = dx_result + x * blockSideLength * dim2, dy_pos = dy_result + x * blockSideLength * dim2;
        {
            y = 0;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_leftCol_block(dim2, block_index, block_dim2, blockSideLength, currBlock, topBlock, bottomBlock, rightBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);  
        }
        for(y=1; y<block_dim2-1; y++){
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_inner_block(dim2, block_index, block_dim2, blockSideLength, currBlock, leftBlock, rightBlock, topBlock, bottomBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);
        }
        {
            y = block_dim2 - 1;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            derivative_process_diff_rightCol_block(dim2, block_index, block_dim2, blockSideLength, currBlock, topBlock, bottomBlock, leftBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);  
        }
    }
    {
        x = block_dim1 - 1;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        dx_pos = dx_result + x * blockSideLength * dim2, dy_pos = dy_result + x * blockSideLength * dim2;
        {
            y = 0;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            derivative_process_diff_sw_corner_block(dim2, block_index, block_dim2, blockSideLength, currBlock, rightBlock, topBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);  
        }
        for(y=1; y<block_dim2-1; y++){
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            derivative_process_diff_bottomRow_block(dim2, block_index, block_dim2, blockSideLength, currBlock, leftBlock, rightBlock, topBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);
        }
        {
            y = block_dim2 - 1;
            block_index = x * block_dim2 + y;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            derivative_process_diff_se_corner_block(dim2, block_index, block_dim2, blockSideLength, currBlock, leftBlock, topBlock, blocks_quant_mean, errorBound, dx_pos, dy_pos);
        }
    }
}

template <class T>
void SZx_derivative_diff_kernel_2dblock(
    unsigned char *cmpData, int *fixedRate, int *offsets,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSideLength,
    T *dx_result, T *dy_result, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * qmean_pos = cmpData + block_num;
    int * currBlock = nullptr, * tempBlockRow = nullptr;
    int * leftBlock = nullptr, * rightBlock = nullptr, * topBlock = nullptr, * bottomBlock = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    std::vector<int> blocks_quant_mean(block_num, 0);
    for(int k=0; k<block_num; k++){
        blocks_quant_mean[k] = (0xff000000 & (*qmean_pos << 24)) |
                                (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                                (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                                (0x000000ff & *(qmean_pos+3));
        qmean_pos += 4;
    }
    size_t prefix_length = 0;
    for(int x=0; x<block_dim1; x++){
        for(int y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate = (int)cmpData[block_index];
            fixedRate[block_index] = temp_fixed_rate;
            size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
            if(temp_fixed_rate) 
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
        offsets[x+1] = prefix_length;
    }
    int * prevBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    int * currBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    int * nextBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    SZx_compute_derivative_diff_2dblock<T>(cmpData, block_dim1, block_dim2, blockSideLength, blocks_quant_mean.data(), offsets, fixedRate,
                                   absQuantDiff, signFlag, prevBlockRow, currBlockRow, nextBlockRow, dx_result, dy_result, errorBound);
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
}

/**
 * dataRowwise-serial
 */
void SZx_recoverToQuant_blockRow(
    int blockRow_ind, size_t dim2, int block_dim2,
    int blockSideLength, int blockSize,
    int *fixedRate, int *blocks_quant_mean,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    unsigned char *cmpData_pos, int *quantInds
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    int y, i, j;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int quant_diff;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            for(i=0; i<blockSideLength; i++){
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    local_index = local_offset + j;
                    global_index = i * dim2 + y * blockSideLength + j;
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    quantInds[global_index] = quant_diff + blocks_quant_mean[block_index];
                }
            }
        }
        else{
            for(i=0; i<blockSideLength; i++){
                int tar_quant = blocks_quant_mean[block_index];
                for(j=0; j<blockSideLength; j++){
                    global_index = i * dim2 + y * blockSideLength + j;
                    quantInds[global_index] = tar_quant;
                }
            }
        }
    }
}                                    

template <class T>
void derivative_process_quant_topDataRow(
    size_t dim2, int *curr_row,
    int *next_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int j;
    {
        j = 0;
        dx_pos[j] = (curr_row[j + 1] - curr_row[j]) * errorBound * 2;
        dy_pos[j] = (next_row[j] - curr_row[j]) * errorBound * 2;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * errorBound;
        dy_pos[j] = (next_row[j] - curr_row[j]) * errorBound * 2;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = (curr_row[j] - curr_row[j - 1]) * errorBound * 2;
        dy_pos[j] = (next_row[j] - curr_row[j]) * errorBound * 2;
    }
}

template <class T>
inline void derivative_process_quant_bottomDataRow(
    size_t dim2, int *curr_row,
    int *prev_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int j;
    {
        j = 0;
        dx_pos[j] = (curr_row[j + 1] - curr_row[j]) * errorBound * 2;
        dy_pos[j] = (curr_row[j] - prev_row[j]) * errorBound * 2;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * errorBound;
        dy_pos[j] = (curr_row[j] - prev_row[j]) * errorBound * 2;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = (curr_row[j] - curr_row[j - 1]) * errorBound * 2;
        dy_pos[j] = (curr_row[j] - prev_row[j]) * errorBound * 2;
    }
}

template <class T>
inline void derivative_process_quant_centralDataRow(
    size_t dim2, int *curr_row,
    int *prev_row, int *next_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int j;
    {
        j = 0;
        dx_pos[j] = (curr_row[j + 1] - curr_row[j]) * errorBound * 2;
        dy_pos[j] = (next_row[j] - prev_row[j]) * errorBound;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * errorBound;
        dy_pos[j] = (next_row[j] - prev_row[j]) * errorBound;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = (curr_row[j] - curr_row[j - 1]) * errorBound * 2;
        dy_pos[j] = (next_row[j] - prev_row[j]) * errorBound;
    }

}

template <class T>
void SZx_compute_derivative_quant_2dblock(
    unsigned char *cmpData, size_t block_dim1, size_t block_dim2,
    int blockSideLength, int *blocks_quant_mean, int *offsets,
    int *fixedRate, unsigned int *absQuantDiff, unsigned char *signFlag,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    T *dx_result, T *dy_result, double errorBound
){
    size_t dim2 = block_dim2 * blockSideLength;
    int blockSize = blockSideLength * blockSideLength;
    int block_num = block_dim1 * block_dim2;
    unsigned char * encode_pos = cmpData + 5 * block_num;
    int *tempBlockRow = nullptr;
    int *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    int x, i;
    int res_offset;
    {
        x = 0;
        res_offset = x * blockSideLength * dim2;
        SZx_recoverToQuant_blockRow(x, dim2, block_dim2, blockSideLength, blockSize, fixedRate, blocks_quant_mean, absQuantDiff, signFlag, encode_pos+offsets[x], currBlockRow);
        SZx_recoverToQuant_blockRow(x+1, dim2, block_dim2, blockSideLength, blockSize, fixedRate, blocks_quant_mean, absQuantDiff, signFlag, encode_pos+offsets[x+1], nextBlockRow);
        {
            i = 0;
            curr_row = currBlockRow + i * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_topDataRow(dim2, curr_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = nextBlockRow;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
    }
    for(x=1; x<block_dim1-1; x++){
        tempBlockRow = prevBlockRow;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        nextBlockRow = tempBlockRow;
        SZx_recoverToQuant_blockRow(x+1, dim2, block_dim2, blockSideLength, blockSize, fixedRate, blocks_quant_mean, absQuantDiff, signFlag, encode_pos+offsets[x+1], nextBlockRow);
        res_offset = x * blockSideLength * dim2;
        {
            i = 0;
            curr_row = currBlockRow + i * dim2;
            prev_row = prevBlockRow + (blockSideLength - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = nextBlockRow;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
    }
    {
        x = block_dim1 - 1;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        res_offset = x * blockSideLength * dim2;
        {
            i = 0;
            curr_row = currBlockRow + i * dim2;
            prev_row = prevBlockRow + (blockSideLength - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_quant_bottomDataRow(dim2, curr_row, prev_row, dx_pos, dy_pos, errorBound);
        }
    }
}

template <class T>
void SZx_derivative_quant_kernel_2dblock(
    unsigned char *cmpData, int *fixedRate, int *offsets,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSideLength,
    T *dx_result, T *dy_result, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * qmean_pos = cmpData + block_num;
    std::vector<int> blocks_quant_mean(block_num, 0);
    for(int k=0; k<block_num; k++){
        blocks_quant_mean[k] = (0xff000000 & (*qmean_pos << 24)) |
                                (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                                (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                                (0x000000ff & *(qmean_pos+3));
        qmean_pos += 4;
    }
    size_t prefix_length = 0;
    for(int x=0; x<block_dim1; x++){
        for(int y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate = (int)cmpData[block_index];
            fixedRate[block_index] = temp_fixed_rate;
            size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
            if(temp_fixed_rate) 
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
        offsets[x+1] = prefix_length;
    }
    int * prevBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    int * currBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    int * nextBlockRow = (int *)malloc(blockSize * block_dim2 * sizeof(int));
    SZx_compute_derivative_quant_2dblock<T>(cmpData, block_dim1, block_dim2, blockSideLength, blocks_quant_mean.data(), offsets, fixedRate,
                                         absQuantDiff, signFlag, prevBlockRow, currBlockRow, nextBlockRow, dx_result, dy_result, errorBound);
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
}

#endif