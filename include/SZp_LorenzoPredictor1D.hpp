#ifndef _SZP_1DLORENZO_PREDICTOR_HPP
#define _SZP_1DLORENZO_PREDICTOR_HPP

#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>
#include "typemanager.hpp"
#include "application_utils.hpp"

inline int quant_diff_predictBy1DLorenzo(int index, float *oriData, double errorBound, int& prev_quant)
{
    int curr_quant = static_cast<int>(std::floor((oriData[index] + errorBound) / (2 * errorBound)));
    int quant_diff = curr_quant - prev_quant;
    prev_quant = curr_quant;
    return quant_diff;
}

void SZp_compress_kernel_1dLorenzo(
    float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    double errorBound, size_t *cmpSize
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    unsigned char * cmpData_pos = cmpData + block_num;
    int x, y, i, j;
    for(x=0; x<block_dim1; x++){
        std::vector<int> ending_quant(blockSideLength, 0);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int global_index, global_offset, local_index;
            int quant_diff, prev_quant, max_quant_diff = 0;
            int temp_fixed_rate;
            for(i=0; i<blockSideLength; i++){
                global_offset = (x * blockSideLength + i) * dim2 + y * blockSideLength;
                prev_quant = ending_quant[i];
                for(j=0; j<blockSideLength; j++){
                    global_index = global_offset + j;
                    local_index = i * blockSideLength + j;
                    quant_diff = quant_diff_predictBy1DLorenzo(global_index, oriData, errorBound, prev_quant);
                    signFlag[local_index] = (quant_diff < 0);
                    absQuantDiff[local_index] = abs(quant_diff);
                    max_quant_diff = max_quant_diff > absQuantDiff[local_index] ? max_quant_diff : absQuantDiff[local_index];
                }
                ending_quant[i] = prev_quant;
            }
            temp_fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
            cmpData[block_index] = (unsigned char)temp_fixed_rate;
            if(temp_fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, blockSize, cmpData_pos);
                cmpData_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absQuantDiff, blockSize, cmpData_pos, temp_fixed_rate);
                cmpData_pos += savedbitsbyteLength;
            }
        }
    }
    *cmpSize = cmpData_pos - cmpData;
}

void SZp_decompress_kernel_1dLorenzo(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * cmpData_pos = cmpData + block_num;
    int x, y, i, j;
    for(x=0; x<block_dim1; x++){
        std::vector<int> ending_quant(blockSideLength, 0);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int global_index, global_offset, local_index;
            int quant_diff, prev_quant, max_quant_diff = 0;
            int temp_fixed_rate = (int)cmpData[block_index];
            if(temp_fixed_rate){
                convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
                cmpData_pos += savedbitsbytelength;
                int curr_quant, quant_diff, prev_quant;
                for(i=0; i<blockSideLength; i++){
                    prev_quant = ending_quant[i];
                    global_offset = (x * blockSideLength + i) * dim2 + y * blockSideLength;
                    for(j=0; j<blockSideLength; j++){
                        global_index = global_offset + j;
                        local_index = i * blockSideLength + j;
                        int sign = -(int)signFlag[local_index];
                        quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                        curr_quant = quant_diff + prev_quant;
                        decData[global_index] = curr_quant * errorBound * 2;
                        prev_quant = curr_quant;
                    }
                    ending_quant[i] = curr_quant;
                }
            }
            else{
                for(i=0; i<blockSideLength; i++){
                    global_offset = (x * blockSideLength + i) * dim2 + y * blockSideLength;
                    int tar_quant = ending_quant[i];
                    for(int j=0; j<blockSideLength; j++){
                        global_index = global_offset + j;
                        decData[global_index] = tar_quant * errorBound * 2;
                    }
                    ending_quant[i] = tar_quant;
                }
            }
        }
    }
}

/**
 * blockwise-serial
 */
void SZp_recoverToQuant_blockRow_1dLorenzo(
    int blockRow_ind, int block_dim2,
    int blockSideLength, int blockSize,
    int *fixedRate, unsigned int *absQuantDiff, unsigned char *signFlag,
    unsigned char *cmpData_pos, int *quantInds
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    std::vector<int> ending_quant(blockSideLength, 0);
    int y, i, j;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            int curr_quant, quant_diff, prev_quant;
            for(i=0; i<blockSideLength; i++){
                prev_quant = ending_quant[i];
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    local_index = local_offset + j;
                    global_index = global_offset + local_index;
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    curr_quant = quant_diff + prev_quant;
                    prev_quant = curr_quant;
                    quantInds[global_index] = curr_quant;
                }
                ending_quant[i] = curr_quant;
            }
        }
        else{
            for(i=0; i<blockSideLength; i++){
                int tar_quant = ending_quant[i];
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    global_index = global_offset + local_offset + j;
                    quantInds[global_index] = tar_quant;
                }
                ending_quant[i] = tar_quant;
            }
        }
    }
}                                    

void SZp_compressFromQuant_blockRow_1dLorenzo(
    int blockRow_ind, int block_dim2,
    int blockSideLength, int blockSize,
    int *fixedRate, unsigned int *absQuantDiff, unsigned char *signFlag,
    unsigned char *cmpData, int *quantInds,
    int *offsets, size_t& prefix_length, size_t& cmpSize
){
    std::vector<int> ending_quant(blockSideLength, 0);
    int y, i, j;
    unsigned char *cmpData_pos = cmpData;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int curr_quant, quant_diff, prev_quant;
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int max_quant_diff = 0;
        for(i=0; i<blockSideLength; i++){
            prev_quant = ending_quant[i];
            local_offset = i * blockSideLength;
            for(j=0; j<blockSideLength; j++){
                local_index = local_offset + j;
                global_index = global_offset + local_index;
                curr_quant = quantInds[global_index];
                quant_diff = curr_quant - prev_quant;
                prev_quant = curr_quant;
                signFlag[local_index] = (quant_diff < 0);
                absQuantDiff[local_index] = abs(quant_diff);
                max_quant_diff = max_quant_diff > absQuantDiff[local_index] ? max_quant_diff : absQuantDiff[local_index];
            }
            ending_quant[i] = curr_quant;
        }
        int temp_fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
        fixedRate[block_index] = (unsigned char)temp_fixed_rate;
        if(temp_fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, blockSize, cmpData_pos);
            cmpData_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absQuantDiff, blockSize, cmpData_pos, temp_fixed_rate);
            cmpData_pos += savedbitsbyteLength;
        }
    }
    size_t increment = cmpData_pos - cmpData;
    cmpSize += increment;
    prefix_length += increment;
    offsets[blockRow_ind] = prefix_length;
}

inline void integerize_quant(
    int left, int right, int top, int bottom,
    int index, int *quantInds
){
    int center = left + right + top + bottom;
    unsigned char sign = (center >> 31) & 1;
    quantInds[index] = (center + (sign ? -2 : 2)) >> 2;
}

inline void heatdis_process_quant_nw_corner_block_edges(
    int q_S, int q_W, int blockSideLength, int *currBlock,
    int *rightBlock, int *bottomBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        top = q_S;
        {
            index = 0;
            left = q_W, right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        left = q_W;
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_ne_corner_block_edges(
    int q_S, int q_W, int blockSideLength, int *currBlock,
    int *leftBlock, int *bottomBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        top = q_S;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        right = q_W;
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_sw_corner_block_edges(
    int q_B, int q_W, int blockSideLength, int *currBlock,
    int *rightBlock, int *topBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = q_W, right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        bottom = q_B;
        {
            index = i * blockSideLength;
            left = q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        left = q_W;
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_se_corner_block_edges(
    int q_B, int q_W, int blockSideLength, int *currBlock,
    int *leftBlock, int *topBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        bottom = q_B;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        right = q_W;
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_topRow_block_edges(
    int q_S, int blockSideLength,
    int *currBlock, int *bottomBlock,
    int *leftBlock, int *rightBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        top = q_S;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_bottomRow_block_edges(
    int q_B, int blockSideLength,
    int *currBlock, int *topBlock,
    int *leftBlock, int *rightBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        bottom = q_B;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_leftCol_block_edges(
    int q_W, int blockSideLength,
    int *currBlock, int *rightBlock,
    int *topBlock, int *bottomBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = q_W, right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_rightCol_block_edges(
    int q_W, int blockSideLength,
    int *currBlock, int *leftBlock,
    int *topBlock, int *bottomBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = q_W, top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = q_W, top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_quant_inner_block_edges(
    int blockSideLength, int *currBlock,
    int *leftBlock, int *rightBlock,
    int *topBlock, int *bottomBlock, int *update
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        // top horizontal edge
        i = 0;
        {
            index = 0;
            left = leftBlock[blockSideLength - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[(blockSideLength-1) * blockSideLength + j], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[0], top = topBlock[blockSideLength * blockSideLength - 1], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // bottom horizontal edge
        i = blockSideLength - 1;
        {
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[0];
            integerize_quant(left, right, top, bottom, index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index++;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_quant(left, right, top, bottom, index, update);
        }
        {
            index++;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[blockSideLength - 1];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // left vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength;
            left = leftBlock[index + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
    {
        // right vertical edge (excluding endpoints)
        for(i=1; i<blockSideLength-1; i++){
            index = i * blockSideLength + blockSideLength - 1;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_quant(left, right, top, bottom, index, update);
        }
    }
}

inline void heatdis_process_block_interior(
    int blockSideLength,
    int *currBlock, int *update
){
    int i, j;
    int index;
    for(i=1; i<blockSideLength-1; i++){
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            integerize_quant(currBlock[index - 1], currBlock[index + 1], currBlock[index - blockSideLength], currBlock[index + blockSideLength], index, update);
        }
    }
}

void heatdis_update_quantInds_1dLorenzo(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absQuantDiff, unsigned char *signFlag, int *updateBlockRow,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    int block_dim1, int block_dim2, int blockSideLength,
    int q_S, int q_W, int q_B,
    int current, int next, int iter, size_t& cmpSize
){
    int blockSize = blockSideLength * blockSideLength;
    int block_num = block_dim1 * block_dim2;
    size_t prefix_length = 0;
    int * update_pos = nullptr, * currBlock = nullptr, * tempBlockRow = nullptr;
    int * leftBlock = nullptr, * rightBlock = nullptr, * topBlock = nullptr, * bottomBlock = nullptr;
    unsigned char * nextBlockRow_cmpData = nullptr;
    unsigned char * cmpData_pos = cmpData[current] + block_num;
    unsigned char * cmpData_pos_update = cmpData[next] + block_num;
    int x, y;
    {
        x = 0;
        SZp_recoverToQuant_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, fixedRate[current], absQuantDiff, signFlag, cmpData_pos, currBlockRow);
        SZp_recoverToQuant_blockRow_1dLorenzo(x+1, block_dim2, blockSideLength, blockSize, fixedRate[current], absQuantDiff, signFlag, cmpData_pos+offsets[current][x], nextBlockRow);
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_nw_corner_block_edges(q_S, q_W, blockSideLength, currBlock, rightBlock, bottomBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        for(y=1; y<block_dim2-1; y++){
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_topRow_block_edges(q_S, blockSideLength, currBlock, bottomBlock, leftBlock, rightBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_ne_corner_block_edges(q_S, q_W, blockSideLength, currBlock, leftBlock, bottomBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        SZp_compressFromQuant_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, fixedRate[next], absQuantDiff, signFlag,
                                                    cmpData_pos_update, updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
    for(x=1; x<block_dim1-1; x++){
        tempBlockRow = prevBlockRow;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        nextBlockRow = tempBlockRow;
        SZp_recoverToQuant_blockRow_1dLorenzo(x+1, block_dim2, blockSideLength, blockSize, fixedRate[current], absQuantDiff, signFlag, cmpData_pos+offsets[current][x], nextBlockRow);
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_leftCol_block_edges(q_W, blockSideLength, currBlock, rightBlock, topBlock, bottomBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        for(y=1; y<block_dim2-1; y++){
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_inner_block_edges(blockSideLength, currBlock, leftBlock, rightBlock, topBlock, bottomBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_rightCol_block_edges(q_W, blockSideLength, currBlock, leftBlock, topBlock, bottomBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        SZp_compressFromQuant_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, fixedRate[next], absQuantDiff, signFlag,
                                                    cmpData_pos_update+offsets[next][x-1], updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
    {
        x = block_dim1 - 1;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_sw_corner_block_edges(q_B, q_W, blockSideLength, currBlock, rightBlock, topBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        for(y=1; y<block_dim2-1; y++){
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_bottomRow_block_edges(q_B, blockSideLength, currBlock, topBlock, leftBlock, rightBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_quant_se_corner_block_edges(q_B, q_W, blockSideLength, currBlock, leftBlock, topBlock, update_pos);
            heatdis_process_block_interior(blockSideLength, currBlock, update_pos);
        }
        SZp_compressFromQuant_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, fixedRate[next], absQuantDiff, signFlag,
                                                    cmpData_pos_update+offsets[next][x-1], updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
}

void SZp_heatdis_kernel_quant_1dLorenzo(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t *cmpSize, int max_iter
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    int current = 0, next = 1;
    size_t prefix_length = 0;
    for(int x=0; x<block_dim1; x++){
        for(int y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate = (int)cmpData[current][block_index];
            fixedRate[current][block_index] = temp_fixed_rate;
            size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
            if(temp_fixed_rate) 
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
        offsets[current][x] = prefix_length;
    }
    const int q_S = static_cast<int>(std::floor((SRC_TEMP + errorBound) / (2 * errorBound)));
    const int q_W = static_cast<int>(std::floor((WALL_TEMP + errorBound) / (2 * errorBound)));
    const int q_B = static_cast<int>(std::floor((BACK_TEMP + errorBound) / (2 * errorBound)));
    int * prevBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * currBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * nextBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * updateBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        heatdis_update_quantInds_1dLorenzo(cmpData, offsets, fixedRate, absQuantDiff, signFlag,
                        updateBlockRow, prevBlockRow, currBlockRow, nextBlockRow,
                        block_dim1, block_dim2, blockSideLength, q_S, q_W, q_B,
                        current, next, iter, compressed_size);
        current = next;
        next = 1 - current;
    }
    *cmpSize = compressed_size;
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
    free(updateBlockRow);
}                                            

void SZp_recoverToLorenzo_blockRow_1dLorenzo(
    int blockRow_ind, int block_dim2,
    int blockSideLength, int blockSize,
    unsigned int *absQuantDiff, unsigned char *signFlag, int *fixedRate,
    unsigned char *cmpData_pos, int *lorenzoPred,
    std::vector<int>& tailofBlockRow
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    int y, i, j;
    memset(tailofBlockRow.data(), 0, sizeof(int)*blockSideLength);
    for(int k=0; k<blockSideLength; k++) tailofBlockRow[k] = 0;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int quant_diff, tail;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            for(i=0; i<blockSideLength; i++){
                local_offset = i * blockSideLength;
                tail = 0;
                for(j=0; j<blockSideLength; j++){
                    local_index = local_offset + j;
                    global_index = global_offset + local_index;
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    lorenzoPred[global_index] = quant_diff;
                    tail += quant_diff;
                }
                tailofBlockRow[i] += tail;
            }
        }
        else{
            for(i=0; i<blockSideLength; i++){
                local_offset = i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    global_index = global_offset + local_offset + j;
                    lorenzoPred[global_index] = 0;
                }
            }
        }
    }
}                                    

void SZp_compressFromLorenzo_blockRow_1dLorenzo(
    int blockRow_ind, int block_dim2,
    int blockSideLength, int blockSize,
    unsigned int *absQuantDiff, unsigned char *signFlag, int *fixedRate, 
    unsigned char *cmpData, int *lorenzoPred,
    int *offsets, size_t& prefix_length, size_t& cmpSize
){
    int y, i, j;
    unsigned char *cmpData_pos = cmpData;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int quant_diff, max_quant_diff = 0;
        for(i=0; i<blockSideLength; i++){
            local_offset = i * blockSideLength;
            for(j=0; j<blockSideLength; j++){
                local_index = local_offset + j;
                global_index = global_offset + local_index;
                quant_diff = lorenzoPred[global_index];
                signFlag[local_index] = (quant_diff < 0);
                absQuantDiff[local_index] = abs(quant_diff);
                max_quant_diff = max_quant_diff > absQuantDiff[local_index] ? max_quant_diff : absQuantDiff[local_index];
            }
        }
        int temp_fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
        fixedRate[block_index] = (unsigned char)temp_fixed_rate;
        if(temp_fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, blockSize, cmpData_pos);
            cmpData_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absQuantDiff, blockSize, cmpData_pos, temp_fixed_rate);
            cmpData_pos += savedbitsbyteLength;
        }
    }
    size_t increment = cmpData_pos - cmpData;
    cmpSize += increment;
    prefix_length += increment;
    offsets[blockRow_ind] = prefix_length;
}

inline void integerize_lorenzo(
    int left, int right, int top, int bottom,
    int bias, int& residual, int index, int *lorenzo
){
    int center = left + right + top + bottom;
    int predict = center + residual + bias;
    lorenzo[index] = predict >> 2;
    residual = (predict & 0x3) - bias;
}

inline void heatdis_process_lorenzo_nw_corner_block(
    int q_S, int q_W, int bias,
    int blockSideLength, int *currBlock,
    int *rightBlock, int *bottomBlock, int *update,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = q_S, bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        top = 0;
        {
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}                                    

inline void heatdis_process_lorenzo_ne_corner_block(
    int q_S, int q_W, int bias,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *bottomBlock, int *update,
    const std::vector<int> tailofBlockRow,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        top = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}                                    

inline void heatdis_process_lorenzo_sw_corner_block(
    int q_B, int q_W, int bias,
    int blockSideLength, int *currBlock,
    int *rightBlock, int *topBlock, int *update,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        residuals[i] = 0;
        int topBlock_offset = (blockSideLength - 1) * blockSideLength;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = q_B;
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        bottom = 0;
        {
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}                                    

inline void heatdis_process_lorenzo_se_corner_block(
    int q_B, int q_W, int bias,
    int blockSideLength, int *currBlock,
    int *leftBlock, int *topBlock, int *update,
    const std::vector<int> tailofBlockRow,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        int topBlock_offset = (blockSideLength - 1) * blockSideLength;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        bottom = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}                                    

inline void heatdis_process_lorenzo_leftCol_block(
    int q_W, int bias, int blockSideLength,
    int *currBlock, int *rightBlock,
    int *topBlock, int *bottomBlock,
    int *update, std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        residuals[i] = 0;
        int topBlock_offset = (blockSideLength - 1) * blockSideLength;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        residuals[i] = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = q_W, right = currBlock[index] + currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
            j = 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1] - q_W, right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=2; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}

inline void heatdis_process_lorenzo_rightCol_block(
    int q_W, int bias, int blockSideLength,
    int *currBlock, int *leftBlock,
    int *topBlock, int *bottomBlock, int *update,
    const std::vector<int> tailofBlockRow,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        int topBlock_offset = (blockSideLength - 1) * blockSideLength;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = q_W - tailofBlockRow[i], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}

inline void heatdis_process_lorenzo_topRow_block(
    int bias, int blockSideLength,
    int *currBlock, int *bottomBlock,
    int *leftBlock, int *rightBlock, int *update,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        top = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}

inline void heatdis_process_lorenzo_bottomRow_block(
    int bias, int blockSideLength,
    int *currBlock, int *topBlock,
    int *leftBlock, int *rightBlock, int *update,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    {
        i = 0;
        int topBlock_offset = (blockSideLength - 1) * blockSideLength;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        bottom = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}

inline void heatdis_process_lorenzo_inner_block(
    int bias, int blockSideLength,
    int *currBlock, int *leftBlock, int *rightBlock,
    int *topBlock, int *bottomBlock, int *update,
    std::vector<int>& residuals
){
    int i, j;
    int left, right, top, bottom;
    int index;
    int topBlock_offset = (blockSideLength - 1) * blockSideLength;
    {
        i = 0;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = topBlock[topBlock_offset + j], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    for(i=1; i<blockSideLength-1; i++){
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = currBlock[index + blockSideLength];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
    {
        i = blockSideLength - 1;
        {
            j = 0;
            index = i * blockSideLength + j;
            left = leftBlock[i * blockSideLength + blockSideLength - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        for(j=1; j<blockSideLength-1; j++){
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = currBlock[index + 1], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
        {
            j = blockSideLength - 1;
            index = i * blockSideLength + j;
            left = currBlock[index - 1], right = rightBlock[i * blockSideLength], top = currBlock[index - blockSideLength], bottom = bottomBlock[j];
            integerize_lorenzo(left, right, top, bottom, bias, residuals[i], index, update);
        }
    }
}

void heatdis_update_lorenzoPred_1dLorenzo(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absQuantDiff, unsigned char *signFlag, int *updateBlockRow,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    int block_dim1, int block_dim2, int blockSideLength,
    int q_S, int q_W, int q_B, size_t& cmpSize,
    int current, int next, int iter
){
    int blockSize = blockSideLength * blockSideLength;
    int block_num = block_dim1 * block_dim2;
    int bias = (iter & 1) + 1;
    size_t prefix_length = 0;
    int * update_pos = nullptr, * currBlock = nullptr, * tempBlockRow = nullptr;
    int * leftBlock = nullptr, * rightBlock = nullptr, * topBlock = nullptr, * bottomBlock = nullptr;
    unsigned char * nextBlockRow_cmpData = nullptr;
    unsigned char * cmpData_pos = cmpData[current] + block_num;
    unsigned char * cmpData_pos_update = cmpData[next] + block_num;
    std::vector<int> tailofCurrBlockRow(blockSideLength, 0);
    std::vector<int> tailofNextBlockRow(blockSideLength, 0);
    std::vector<int> residuals(blockSideLength, 0);
    int x, y;
    {
        x = 0;
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[current], cmpData_pos, currBlockRow, tailofCurrBlockRow);
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x+1, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[current], cmpData_pos+offsets[current][x], nextBlockRow, tailofNextBlockRow);
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_nw_corner_block(q_S, q_W, bias, blockSideLength, currBlock, rightBlock, bottomBlock, update_pos, residuals);            
        }
        for(y=1; y<block_dim2-1; y++){
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_topRow_block(bias, blockSideLength, currBlock, bottomBlock, leftBlock, rightBlock, update_pos, residuals);
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_ne_corner_block(q_S, q_W, bias, blockSideLength, currBlock, leftBlock, bottomBlock, update_pos, tailofCurrBlockRow, residuals);
        }
        SZp_compressFromLorenzo_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[next],
                                                    cmpData_pos_update, updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
    for(x=1; x<block_dim1-1; x++){
        tempBlockRow = prevBlockRow;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        nextBlockRow = tempBlockRow;
        memcpy(tailofCurrBlockRow.data(), tailofNextBlockRow.data(), sizeof(int)*blockSideLength);
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x+1, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[current], cmpData_pos+offsets[current][x], nextBlockRow, tailofNextBlockRow);
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_leftCol_block(q_W, bias, blockSideLength, currBlock, rightBlock, topBlock, bottomBlock, update_pos, residuals);
        }
        for(y=1; y<block_dim2-1; y++){
            // {
            //     currBlock = currBlockRow + y * blockSize;
            //     leftBlock = currBlockRow + (y - 1) * blockSize;
            //     rightBlock = currBlockRow + (y + 1) * blockSize;
            //     topBlock = prevBlockRow + y * blockSize;
            //     bottomBlock = nextBlockRow + y * blockSize;
            //     update_pos = updateBlockRow + y * blockSize;
            //     heatdis_process_lorenzo_inner_block(bias, blockSideLength, currBlock, leftBlock, rightBlock, topBlock, bottomBlock, update_pos, residuals);
            // }
            {
                bool do_simplify = true;
                int centerIndex = x * block_dim2 + y;
                if(fixedRate[current][centerIndex]) do_simplify = false;
                if(fixedRate[current][centerIndex-1]) do_simplify = false;
                if(fixedRate[current][centerIndex+1]) do_simplify = false;
                if(fixedRate[current][centerIndex-block_dim2]) do_simplify = false;
                if(fixedRate[current][centerIndex+block_dim2]) do_simplify = false;
                if(!do_simplify){
                    currBlock = currBlockRow + y * blockSize;
                    leftBlock = currBlockRow + (y - 1) * blockSize;
                    rightBlock = currBlockRow + (y + 1) * blockSize;
                    topBlock = prevBlockRow + y * blockSize;
                    bottomBlock = nextBlockRow + y * blockSize;
                    update_pos = updateBlockRow + y * blockSize;
                    heatdis_process_lorenzo_inner_block(bias, blockSideLength, currBlock, leftBlock, rightBlock, topBlock, bottomBlock, update_pos, residuals);
                }
                else{
                    update_pos = updateBlockRow + y * blockSize;
                    for(int k=0; k<blockSize; k++) update_pos[k] = 0;
                }
            }
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            bottomBlock = nextBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_rightCol_block(q_W, bias, blockSideLength, currBlock, leftBlock, topBlock, bottomBlock, update_pos, tailofCurrBlockRow, residuals);
        }
        SZp_compressFromLorenzo_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[next],
                                                    cmpData_pos_update+offsets[next][x-1], updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
    {
        x = block_dim1 - 1;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        memcpy(tailofCurrBlockRow.data(), tailofNextBlockRow.data(), sizeof(int)*blockSideLength);
        {
            y = 0;
            currBlock = currBlockRow + y * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_sw_corner_block(q_B, q_W, bias, blockSideLength, currBlock, rightBlock, topBlock, update_pos, residuals);
        }
        for(y=1; y<block_dim2-1; y++){
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            rightBlock = currBlockRow + (y + 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_bottomRow_block(bias, blockSideLength, currBlock, topBlock, leftBlock, rightBlock, update_pos, residuals);
        }
        {
            y = block_dim2 - 1;
            currBlock = currBlockRow + y * blockSize;
            leftBlock = currBlockRow + (y - 1) * blockSize;
            topBlock = prevBlockRow + y * blockSize;
            update_pos = updateBlockRow + y * blockSize;
            heatdis_process_lorenzo_se_corner_block(q_B, q_W, bias, blockSideLength, currBlock, leftBlock, topBlock, update_pos, tailofCurrBlockRow, residuals);
        }
        SZp_compressFromLorenzo_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate[next],
                                                    cmpData_pos_update+offsets[next][x-1], updateBlockRow, offsets[next], prefix_length, cmpSize);
        for(y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            cmpData[next][block_index] = (unsigned char)fixedRate[next][block_index];
        }
    }
}                        

void SZp_heatdis_kernel_lorenzo_1dLorenzo(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t *cmpSize, int max_iter
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    int current = 0, next = 1;
    size_t prefix_length = 0;
    for(int x=0; x<block_dim1; x++){
        for(int y=0; y<block_dim2; y++){
            int block_index = x * block_dim2 + y;
            int temp_fixed_rate = (int)cmpData[current][block_index];
            fixedRate[current][block_index] = temp_fixed_rate;
            size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
            if(temp_fixed_rate) 
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
        offsets[current][x] = prefix_length;
    }
    const int q_S = static_cast<int>(std::floor((SRC_TEMP + errorBound) / (2 * errorBound)));
    const int q_W = static_cast<int>(std::floor((WALL_TEMP + errorBound) / (2 * errorBound)));
    const int q_B = static_cast<int>(std::floor((BACK_TEMP + errorBound) / (2 * errorBound)));
    int * prevBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * currBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * nextBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * updateBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        heatdis_update_lorenzoPred_1dLorenzo(cmpData, offsets, fixedRate, absQuantDiff, signFlag,
                            updateBlockRow, prevBlockRow, currBlockRow, nextBlockRow,
                            block_dim1, block_dim2, blockSideLength, q_S, q_W, q_B, compressed_size,
                            current, next, iter);
        current = next;
        next = 1 - current;
    }
    *cmpSize = compressed_size;
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
    free(updateBlockRow);
}                                            

long int SZp_mean_kernel_quant_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    int *quantInds, int blockSideLength, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * cmpData_pos = cmpData + block_num;
    std::vector<int> fixedRate(block_num, 0);
    std::vector<int> offsets(block_dim1+1, 0);
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
    int x, i;
    long int quant_sum = 0;
    int buffer_size = blockSideLength * dim2;
    for(x=0; x<block_dim1; x++){
        SZp_recoverToQuant_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, fixedRate.data(), absQuantDiff, signFlag, cmpData_pos+offsets[x], quantInds);
        for(i=0; i<buffer_size; i++){
            quant_sum += quantInds[i];
        }
    }
    return quant_sum;
}

long int SZp_mean_kernel_lorenzo_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    int *lorenzoPred, int blockSideLength, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * cmpData_pos = cmpData + block_num;
    std::vector<int> fixedRate(block_num, 0);
    std::vector<int> offsets(block_dim1+1, 0);
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
    int x, y, i, j;
    long int quant_sum = 0;
    for(x=0; x<block_dim1; x++){
        std::vector<int> tailofBlockRow(blockSideLength, 0);
        std::vector<int> prefix_sum(blockSideLength, 0);
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate.data(), cmpData_pos+offsets[x], lorenzoPred, tailofBlockRow);
        for(y=0; y<block_dim2; y++){
            int * curr_block = lorenzoPred + y * blockSize;
            for(i=0; i<blockSideLength; i++){
                quant_sum += prefix_sum[i] * blockSideLength;
                int * curr_row = curr_block + i * blockSideLength;
                for(j=0; j<blockSideLength; j++){
                    prefix_sum[i] += curr_row[j];
                    quant_sum += (blockSideLength - j) * curr_row[j];
                }
            }
        }
    }
    return quant_sum;
}

/**
 * dataRowwise-serial
 */
void SZp_recoverToQuant_blockRow_1dLorenzo(
    int blockRow_ind, size_t dim2, int block_dim2,
    int blockSideLength, int blockSize,
    int *fixedRate, unsigned int *absQuantDiff, unsigned char *signFlag,
    unsigned char *cmpData_pos, int *quantInds
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    std::vector<int> ending_quant(blockSideLength, 0);
    int y, i, j;
    int * recover_pos = quantInds;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_index, global_index;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            int curr_quant, quant_diff, prev_quant;
            int * row_pos = recover_pos;
            local_index = 0;
            for(i=0; i<blockSideLength; i++){
                prev_quant = ending_quant[i];
                int * col_pos = row_pos;
                for(j=0; j<blockSideLength; j++){
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    local_index++;
                    curr_quant = quant_diff + prev_quant;
                    prev_quant = curr_quant;
                    *(col_pos++) = curr_quant;
                    // global_index = i * dim2 + y * blockSideLength + j;
                    // quantInds[global_index] = curr_quant;
                }
                ending_quant[i] = curr_quant;
                row_pos += dim2;
            }
        }
        else{
            int * row_pos = recover_pos;
            for(i=0; i<blockSideLength; i++){
                int tar_quant = ending_quant[i];
                int * col_pos = row_pos;
                for(j=0; j<blockSideLength; j++){
                    *(col_pos++) = tar_quant;
                    // global_index = i * dim2 + y * blockSideLength + j;
                    // quantInds[global_index] = tar_quant;
                }
                ending_quant[i] = tar_quant;
                row_pos += dim2;
            }
        }
        recover_pos += blockSideLength;
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
void SZp_derivative_kernel_quant_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    T *dx_result, T *dy_result,
    int blockSideLength, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * cmpData_pos = cmpData + block_num;
    int *tempBlockRow = nullptr;
    int *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    std::vector<int> fixedRate(block_num, 0);
    std::vector<int> offsets(block_dim1+1, 0);
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
    int x, i;
    int res_offset;
    {
        x = 0;
        res_offset = x * blockSideLength * dim2;
        SZp_recoverToQuant_blockRow_1dLorenzo(x, dim2, block_dim2, blockSideLength, blockSize, fixedRate.data(), absQuantDiff, signFlag, cmpData_pos+offsets[x], currBlockRow);
        SZp_recoverToQuant_blockRow_1dLorenzo(x+1, dim2, block_dim2, blockSideLength, blockSize, fixedRate.data(), absQuantDiff, signFlag, cmpData_pos+offsets[x+1], nextBlockRow);
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
        SZp_recoverToQuant_blockRow_1dLorenzo(x+1, dim2, block_dim2, blockSideLength, blockSize, fixedRate.data(), absQuantDiff, signFlag, cmpData_pos+offsets[x+1], nextBlockRow);
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
void SZp_derivative_quant_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    T *dx_result, T *dy_result
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int * prevBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * currBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * nextBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    SZp_derivative_kernel_quant_1dLorenzo<T>(cmpData, dim1, dim2, absQuantDiff, signFlag, prevBlockRow, currBlockRow, nextBlockRow,
                                            dx_result, dy_result, blockSideLength, errorBound);
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
}

/**
 * dataRowwise-serial
 */
void SZp_recoverToLorenzo_blockRow_1dLorenzo(
    int blockRow_ind, size_t dim2, int block_dim2,
    int blockSideLength, int blockSize,
    unsigned int *absQuantDiff, unsigned char *signFlag, int *fixedRate,
    unsigned char *cmpData_pos, int *lorenzoPred
){
    size_t cmp_block_sign_length = (blockSize + 7) / 8;
    int y, i, j;
    int * recover_pos = lorenzoPred;
    for(y=0; y<block_dim2; y++){
        int block_index = blockRow_ind * block_dim2 + y;
        int temp_fixed_rate = fixedRate[block_index];
        int global_offset = blockSize * y;
        int local_offset, local_index, global_index;
        int quant_diff, tail;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absQuantDiff, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            int * row_pos = recover_pos;
            local_index = 0;
            for(i=0; i<blockSideLength; i++){
                // local_offset = i * blockSideLength;
                tail = 0;
                int * col_pos = row_pos;
                for(j=0; j<blockSideLength; j++){
                    // local_index = local_offset + j;
                    int sign = -(int)signFlag[local_index];
                    quant_diff = (absQuantDiff[local_index] ^ sign) - sign;
                    local_index++;
                    // global_index = i * dim2 + y * blockSideLength + j;
                    // lorenzoPred[global_index] = quant_diff;
                    *(col_pos++) = quant_diff;
                    tail += quant_diff;
                }
                row_pos += dim2;
            }
        }
        else{
            int * row_pos = recover_pos;
            for(i=0; i<blockSideLength; i++){
                int * col_pos = row_pos;
                for(j=0; j<blockSideLength; j++){
                    *(col_pos++) = 0;
                    // global_index = i * dim2 + y * blockSideLength + j;
                    // lorenzoPred[global_index] = 0;
                }
                row_pos += dim2;
            }
        }
        recover_pos += blockSideLength;
    }
}                                    

template <class T>
inline void derivative_process_lorenzo_topDataRow(
    size_t dim2, int *curr_row,
    int *next_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int prefix_sum = 0;
    int j;
    {
        j = 0;
        dx_pos[j] = curr_row[j + 1] * errorBound * 2;
        prefix_sum += next_row[j] - curr_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j] + curr_row[j + 1]) * errorBound;
        prefix_sum += next_row[j] - curr_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = curr_row[j] * errorBound * 2;
        prefix_sum += next_row[j] - curr_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
}

template <class T>
inline void derivative_process_lorenzo_bottomDataRow(
    size_t dim2, int *curr_row,
    int *prev_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int prefix_sum = 0;
    int j;
    {
        j = 0;
        dx_pos[j] = curr_row[j + 1] * errorBound * 2;
        prefix_sum += curr_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j] + curr_row[j + 1]) * errorBound;
        prefix_sum += curr_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = curr_row[j] * errorBound * 2;
        prefix_sum += curr_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound * 2;
    }
}

template <class T>
inline void derivative_process_lorenzo_centralDataRow(
    size_t dim2, int *curr_row,
    int *prev_row, int *next_row,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    int prefix_sum = 0;
    int j;
    {
        j = 0;
        dx_pos[j] = curr_row[j + 1] * errorBound * 2;
        prefix_sum += next_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound;
    }
    for(j=1; j<dim2-1; j++){
        dx_pos[j] = (curr_row[j] + curr_row[j + 1]) * errorBound;
        prefix_sum += next_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound;
    }
    {
        j = dim2 - 1;
        dx_pos[j] = curr_row[j] * errorBound * 2;
        prefix_sum += next_row[j] - prev_row[j];
        dy_pos[j] = prefix_sum * errorBound;
    }

}

template <class T>
void SZp_derivative_kernel_lorenzo_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    int *prevBlockRow, int *currBlockRow, int *nextBlockRow,
    T *dx_result, T *dy_result,
    int blockSideLength, double errorBound
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    unsigned char * cmpData_pos = cmpData + block_num;
    int *tempBlockRow = nullptr;
    int *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    std::vector<int> fixedRate(block_num, 0);
    std::vector<int> offsets(block_dim1+1, 0);
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
    int x, i;
    int res_offset;
    {
        x = 0;
        res_offset = x * blockSideLength * dim2;
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x, dim2, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate.data(), cmpData_pos+offsets[x], currBlockRow);
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x+1, dim2, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate.data(), cmpData_pos+offsets[x+1], nextBlockRow);
        {
            i = 0;
            curr_row = currBlockRow + i * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_topDataRow(dim2, curr_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = nextBlockRow;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
    }
    for(x=1; x<block_dim1-1; x++){
        tempBlockRow = prevBlockRow;
        prevBlockRow = currBlockRow;
        currBlockRow = nextBlockRow;
        nextBlockRow = tempBlockRow;
        SZp_recoverToLorenzo_blockRow_1dLorenzo(x+1, dim2, block_dim2, blockSideLength, blockSize, absQuantDiff, signFlag, fixedRate.data(), cmpData_pos+offsets[x+1], nextBlockRow);
        res_offset = x * blockSideLength * dim2;
        {
            i = 0;
            curr_row = currBlockRow + i * dim2;
            prev_row = prevBlockRow + (blockSideLength - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = nextBlockRow;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
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
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        for(i=1; i<blockSideLength-1; i++){
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            next_row = currBlockRow + (i + 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_centralDataRow(dim2, curr_row, prev_row, next_row, dx_pos, dy_pos, errorBound);
        }
        {
            i = blockSideLength - 1;
            curr_row = currBlockRow + i * dim2;
            prev_row = currBlockRow + (i - 1) * dim2;
            dx_pos = dx_result + res_offset + i * dim2;
            dy_pos = dy_result + res_offset + i * dim2;
            derivative_process_lorenzo_bottomDataRow(dim2, curr_row, prev_row, dx_pos, dy_pos, errorBound);
        }
    }
}

template <class T>
void SZp_derivative_lorenzo_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    unsigned int *absQuantDiff, unsigned char *signFlag,
    T *dx_result, T *dy_result
){
    int block_dim1 = (dim1 - 1) / blockSideLength + 1;
    int block_dim2 = (dim2 - 1) / blockSideLength + 1;
    int block_num = block_dim1 * block_dim2;
    int blockSize = blockSideLength * blockSideLength;
    int * prevBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * currBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    int * nextBlockRow = (int *)calloc(blockSize * block_dim2, sizeof(int));
    SZp_derivative_kernel_lorenzo_1dLorenzo<T>(cmpData, dim1, dim2, absQuantDiff, signFlag, prevBlockRow, currBlockRow, nextBlockRow,
                                               dx_result, dy_result, blockSideLength, errorBound);
    free(prevBlockRow);
    free(currBlockRow);
    free(nextBlockRow);
}

#endif