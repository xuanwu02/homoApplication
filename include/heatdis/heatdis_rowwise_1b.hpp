#ifndef _HEATDIS_ROWWISE_1B_HPP
#define _HEATDIS_ROWWISE_1B_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "SZp_typemanager.hpp"
#include "heatdis_utils.hpp"

void SZp_compress_kernel_rowwise_1d_block(
    float *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2,
    unsigned int *absLorenzo, unsigned char *signFlag,
    double errorBound, int blockSize, size_t *cmpSize
){
    int block_num = dim1;
    unsigned char * cmpData_pos = cmpData + block_num;
    for(int x=0; x<dim1; x++){
        int offset = x * dim2;
        int lorenzo_pred, prev_quant = 0;
        int max_lorenzo = 0;
        int temp_fixed_rate;
        int temp_outlier;
        int j, index;
        {   
            j = 0;
            index = offset + j;
            lorenzo_pred = integerize_vanilla(index, oriData, errorBound, prev_quant);
            signFlag[j] = (lorenzo_pred < 0);
            absLorenzo[j] = abs(lorenzo_pred);
            temp_outlier = lorenzo_pred;
            if(temp_outlier < 0) temp_outlier = 0x80000000 | abs(temp_outlier);
            for(int k=3; k>=0; k--){
                *(cmpData_pos++) = (temp_outlier >> (8 * k)) & 0xff;
            }
        }
        for(j=1; j<blockSize; j++){
            index = offset + j;
            lorenzo_pred = integerize_vanilla(index, oriData, errorBound, prev_quant);
            signFlag[j] = (lorenzo_pred < 0);
            absLorenzo[j] = abs(lorenzo_pred);
            max_lorenzo = max_lorenzo > absLorenzo[j] ? max_lorenzo : absLorenzo[j];
        }
        temp_fixed_rate = max_lorenzo == 0 ? 0 : INT_BITS - __builtin_clz(max_lorenzo);
        cmpData[x] = (unsigned char)temp_fixed_rate;
        if(temp_fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, blockSize, cmpData_pos);
            cmpData_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absLorenzo, blockSize, cmpData_pos, temp_fixed_rate);
            cmpData_pos += savedbitsbyteLength;
        }
    }
    *cmpSize = cmpData_pos - cmpData;
}

void SZp_decompress_kernel_rowwise_1d_block(
    float *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2,
    unsigned int *absLorenzo, unsigned char *signFlag,
    double errorBound, int blockSize
){
    int block_num = dim1;
    unsigned char * cmpData_pos = cmpData + block_num;
    size_t cmp_block_sign_length;
    for(int x=0; x<dim1; x++){
        int offset = x * dim2;
        int temp_fixed_rate = (int)cmpData[x];
        int temp_outlier = (0xff000000 & (*cmpData_pos << 24)) |
                            (0x00ff0000 & (*(cmpData_pos+1) << 16)) |
                            (0x0000ff00 & (*(cmpData_pos+2) << 8)) |
                            (0x000000ff & *(cmpData_pos+3));
        cmpData_pos += 4;
        cmp_block_sign_length = (blockSize + 7) / 8;
        int index;
        if(temp_fixed_rate){
            convertByteArray2IntArray_fast_1b_args(blockSize, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, blockSize, absLorenzo, temp_fixed_rate);
            cmpData_pos += savedbitsbytelength;
            absLorenzo[0] = temp_outlier & 0x7fffffff;
            int curr_quant, lorenzo_pred, prev_quant = 0;
            for(int j=0; j<blockSize; j++){
                index = offset + j;
                int sign = -(int)signFlag[j];
                lorenzo_pred = (absLorenzo[j] ^ sign) - sign;
                curr_quant = lorenzo_pred + prev_quant;
                decData[index] = curr_quant * errorBound * 2;
                prev_quant = curr_quant;
            }
        }
        else{
            int first_sign = -((temp_outlier >> 31) & 1);
            int tar_quant = 0;
            if(first_sign)
                tar_quant = (temp_outlier & 0x7fffffff) * -1;
            else
                tar_quant = (temp_outlier & 0x7fffffff);
            if(!temp_outlier) tar_quant = 0;
            for(int j=0; j<blockSize; j++){
                index = offset + j;
                decData[index] = tar_quant * errorBound * 2;
            }
        }
    }
}

void decompressToQuant_rowwise_1d_block(
    int curr_block, int block_size,
    unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
    unsigned char *cmpData_pos, int *quantInds
){
    int temp_outlier = (0xff000000 & (*cmpData_pos << 24)) |
                        (0x00ff0000 & (*(cmpData_pos+1) << 16)) |
                        (0x0000ff00 & (*(cmpData_pos+2) << 8)) |
                        (0x000000ff & *(cmpData_pos+3));
    cmpData_pos += 4;
    int temp_fixed_rate = (int)fixedRate[curr_block];
    int cmp_block_sign_length = (block_size + 7) / 8;
    if(temp_fixed_rate){
        convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
        cmpData_pos += cmp_block_sign_length;
        unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absLorenzo, temp_fixed_rate);
        absLorenzo[0] = temp_outlier & 0x7fffffff;
        int curr_quant, lorenzo_pred, prev_quant = 0;
        for(int j=0; j<block_size; j++){
            int sign = -(int)signFlag[j];
            lorenzo_pred = (absLorenzo[j] ^ sign) - sign;
            curr_quant = lorenzo_pred + prev_quant;
            quantInds[j] = curr_quant;
            prev_quant = curr_quant; 
        }
    }
    else if(!temp_fixed_rate){
        int first_sign = (temp_outlier >> 31) & 1;
        int tar_quant = 0;
        if(first_sign)
            tar_quant = (temp_outlier & 0x7fffffff) * -1;
        else
            tar_quant = (temp_outlier & 0x7fffffff);
        if(!temp_outlier) tar_quant = 0;
        for(int j=0; j<block_size; j++){
            quantInds[j] = tar_quant;
        }
    }
}

void compressFromQuant_rowwise_1d_block(
    int curr_block, int block_size,
    size_t *prefix_length, size_t& cmpSize,
    unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
    int *new_offsets, unsigned char *cmpData_pos, int *quantInds
){
    int lorenzo_pred, curr_quant = 0, prev_quant = 0;
    int temp_fixed_rate, temp_outlier, max_lorenzo = 0;
    for(int j=0; j<block_size; j++){
        curr_quant = quantInds[j];
        lorenzo_pred = curr_quant - prev_quant;
        prev_quant = curr_quant;
        signFlag[j] = (lorenzo_pred < 0);
        absLorenzo[j] = abs(lorenzo_pred);
        if(j != 0)
            max_lorenzo = max_lorenzo > absLorenzo[j] ? max_lorenzo : absLorenzo[j];
        else
            temp_outlier = lorenzo_pred;
    }
    temp_fixed_rate = max_lorenzo == 0 ? 0 : INT_BITS - __builtin_clz(max_lorenzo);
    fixedRate[curr_block] = (unsigned char)temp_fixed_rate;
    if(temp_outlier < 0){
        temp_outlier = 0x80000000 | abs(temp_outlier);
    }
    for(int k=3; k>=0; k--){
        *(cmpData_pos++) = (temp_outlier >> (8 * k)) & 0xff;
    }
    size_t length = 4;
    if(temp_fixed_rate){
        unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
        cmpData_pos += signbyteLength;
        length += signbyteLength;
        unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absLorenzo, block_size, cmpData_pos, temp_fixed_rate);
        cmpData_pos += savedbitsbyteLength;
        length += savedbitsbyteLength;
    }
    *prefix_length += length;
    cmpSize += length;
    new_offsets[curr_block] = *prefix_length;
}

inline void integerize_quant(
    int center, int index,
    int *quantInds
){
    unsigned char sign = (center >> 31) & 1;
    quantInds[index] = (center + (sign ? -2 : 2)) >> 2;
}

inline void process_internal_quant_rowwise_1d_block(
    int blockSize, int *fixedRate_pos,
    int *prevRow, int *currRow, int *nextRow,
    int *update_quant
){
    unsigned char prev = (fixedRate_pos[0] > 0);
    unsigned char curr = (fixedRate_pos[1] > 0);
    unsigned char next = (fixedRate_pos[2] > 0);
    unsigned char type = prev | (curr << 1) | (next << 2);
    int prevRow0 = prevRow[0], currRow0 = currRow[0], nextRow0 = nextRow[0];
    int tmp;
    switch (type)
    {
    case 0:
        tmp = prevRow0 + nextRow0 + currRow0 * 2;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(tmp, j, update_quant);
        }
        break;
    case 1:
        tmp = nextRow0 + currRow0 * 2;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(prevRow[j] + tmp, j, update_quant);
        }
        break;
    case 2:
        tmp = prevRow0 + nextRow0;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(currRow[j-1] + currRow[j+1] + tmp, j, update_quant);
        }
        break;
    case 3:
        tmp = nextRow0;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(prevRow[j] + currRow[j-1] + currRow[j+1] + tmp, j, update_quant);
        }
        break;
    case 4:
        tmp = prevRow0 + currRow0 * 2;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(nextRow[j] + tmp, j, update_quant);
        }
        break;
    case 5:
        tmp = currRow0 * 2;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(prevRow[j] + nextRow[j] + tmp, j, update_quant);
        }
        break;
    case 6:
        tmp = prevRow0;
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(nextRow[j] + currRow[j-1] + currRow[j+1] + tmp, j, update_quant);
        }
        break;
    case 7:
        for(int j=1; j<blockSize-1; j++){
            integerize_quant(prevRow[j] + nextRow[j] + currRow[j-1] + currRow[j+1], j, update_quant);
        }
        break;
    } 
}

void update_quantInds_rowwise_1d_block(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absLorenzo, unsigned char *signFlag, int *update_quant,
    int *prevRow_quant, int *currRow_quant, int *nextRow_quant,
    size_t dim1, size_t dim2, int blockSize, size_t& cmpSize,
    int q_S, int q_W, int q_B,
    int current, int next, int iter
){
    int x, j;
    int center;
    size_t prefix_length = 0;
    int block_num = dim1;
    int * temp = NULL;
    unsigned char * nextRow_cmpData = NULL;
    unsigned char * cmpData_pos = cmpData[current] + block_num;
    unsigned char * cmpData_pos_update = cmpData[next] + block_num;
    // first row
    {
        x = 0;
        decompressToQuant_rowwise_1d_block(x, blockSize, absLorenzo, signFlag, fixedRate[current], cmpData_pos, currRow_quant);
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decompressToQuant_rowwise_1d_block(x + 1, blockSize, absLorenzo, signFlag, fixedRate[current], nextRow_cmpData, nextRow_quant);
        j = 0;
        center = q_W + currRow_quant[j+1] + q_S + nextRow_quant[j];
        integerize_quant(center, j, update_quant);
        for(j=1; j<blockSize-1; j++){
            center = currRow_quant[j-1] + currRow_quant[j+1] + q_S + nextRow_quant[j];
            integerize_quant(center, j, update_quant);
        }
        j = blockSize - 1;
        center = currRow_quant[j-1] + q_W + q_S + nextRow_quant[j];
        integerize_quant(center, j, update_quant);
        compressFromQuant_rowwise_1d_block(x, blockSize, &prefix_length, cmpSize, absLorenzo, signFlag, fixedRate[next],
                                          offsets[next], cmpData_pos_update, update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // row 1 ~ (dim1-2)
    for(x=1; x<dim1-1; x++){
        temp = prevRow_quant;
        prevRow_quant = currRow_quant;
        currRow_quant = nextRow_quant;
        nextRow_quant = temp;
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decompressToQuant_rowwise_1d_block(x + 1, blockSize, absLorenzo, signFlag, fixedRate[current], nextRow_cmpData, nextRow_quant);
        j = 0;
        center = q_W + currRow_quant[j+1] + prevRow_quant[j] + nextRow_quant[j];
        integerize_quant(center, j, update_quant);
        // for(j=1; j<blockSize-1; j++){
        //     center = currRow_quant[j-1] + currRow_quant[j+1] + prevRow_quant[j] + nextRow_quant[j];
        //     integerize_quant(center, j, update_quant);
        // }
        process_internal_quant_rowwise_1d_block(blockSize, fixedRate[current]+x-1, prevRow_quant, currRow_quant, nextRow_quant, update_quant);
        j = blockSize - 1;
        center = currRow_quant[j-1] + q_W + prevRow_quant[j] + nextRow_quant[j];
        integerize_quant(center, j, update_quant);
        compressFromQuant_rowwise_1d_block(x, blockSize, &prefix_length, cmpSize, absLorenzo, signFlag, fixedRate[next],
                                          offsets[next], cmpData_pos_update+offsets[next][x-1], update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // last row
    {
        x = dim1 - 1;
        prevRow_quant = currRow_quant;
        currRow_quant = nextRow_quant;
        j = 0;
        center = q_W + currRow_quant[j+1] + prevRow_quant[j] + q_B;
        integerize_quant(center, j, update_quant);
        for(j=1; j<blockSize-1; j++){
            center = currRow_quant[j-1] + currRow_quant[j+1] + prevRow_quant[j] + q_B;
            integerize_quant(center, j, update_quant);
        }
        j = blockSize - 1;
        center = currRow_quant[j-1] + q_W + prevRow_quant[j] + q_B;
        integerize_quant(center, j, update_quant);
        compressFromQuant_rowwise_1d_block(x, blockSize, &prefix_length, cmpSize, absLorenzo, signFlag, fixedRate[next],
                                          offsets[next], cmpData_pos_update+offsets[next][x-1], update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
}

void SZp_heatdis_kernel_quant_rowwise_1d_block(
    unsigned char **cmpData, int **& offsets, int **& fixedRate,
    unsigned int *absLorenzo, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound, size_t *cmpSize, int max_iter
){
    int block_num = dim1;
    int curr_block, temp_fixed_rate;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    int current = 0, next = 1;
    size_t prefix_length = 0;
    for(int x=0; x<dim1; x++){
        curr_block = x;
        temp_fixed_rate = (int)cmpData[current][curr_block];
        fixedRate[current][curr_block] = temp_fixed_rate;
        size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
        prefix_length += 4;
        if(temp_fixed_rate) 
            prefix_length += cmp_block_sign_length + savedbitsbytelength;
        offsets[current][curr_block] = prefix_length;
    }
    const int q_S = static_cast<int>(std::floor((SRC_TEMP + errorBound) / (2 * errorBound)));
    const int q_W = static_cast<int>(std::floor((WALL_TEMP + errorBound) / (2 * errorBound)));
    const int q_B = static_cast<int>(std::floor((BACK_TEMP + errorBound) / (2 * errorBound)));
    int * prevRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * currRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * nextRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * update_quant = (int *)calloc(blockSize, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        update_quantInds_rowwise_1d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag,
                                        update_quant, prevRow_quant, currRow_quant, nextRow_quant,
                                        dim1, dim2, blockSize, compressed_size,q_S, q_W, q_B, current, next, iter);
        current = next;
        next = 1 - current;
    }
    *cmpSize = compressed_size;
    free(prevRow_quant);
    free(currRow_quant);
    free(nextRow_quant);
    free(update_quant);
}

void decompressToLorenzo_rowwise_1d_block(
    int curr_block, int block_size,
    unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
    unsigned char *cmpData_pos, int *lorenzoPred,
    int *second_quant, int *last_quant
){
    int temp_outlier = (0xff000000 & (*cmpData_pos << 24)) |
                        (0x00ff0000 & (*(cmpData_pos+1) << 16)) |
                        (0x0000ff00 & (*(cmpData_pos+2) << 8)) |
                        (0x000000ff & *(cmpData_pos+3));
    cmpData_pos += 4;
    int temp_fixed_rate = (int)fixedRate[curr_block];
    int cmp_block_sign_length = (block_size + 7) / 8;
    *second_quant = 0;
    *last_quant = 0;
    if(temp_fixed_rate){
        convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
        cmpData_pos += cmp_block_sign_length;
        unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absLorenzo, temp_fixed_rate);
        absLorenzo[0] = temp_outlier & 0x7fffffff;
        int lorenzo;
        for(int j=0; j<block_size; j++){
            int sign = -(int)signFlag[j];
            lorenzo = (absLorenzo[j] ^ sign) - sign;
            *last_quant += lorenzo;
            lorenzoPred[j] = lorenzo;
        }
        *second_quant = lorenzoPred[0] + lorenzoPred[1];
    }
    else if(!temp_fixed_rate){
        int first_sign = (temp_outlier >> 31) & 1;
        int tar_quant = 0;
        if(first_sign)
            tar_quant = (temp_outlier & 0x7fffffff) * -1;
        else
            tar_quant = (temp_outlier & 0x7fffffff);
        if(!temp_outlier) tar_quant = 0;
        lorenzoPred[0] = tar_quant;
        for(int j=1; j<block_size; j++){
            lorenzoPred[j] = 0;
        }
        *second_quant = tar_quant;
        *last_quant = tar_quant;
    }
}

void compressFromLorenzo_rowwise_1d_block(
    int curr_block, int block_size,
    unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
    int *new_offsets, unsigned char *cmpData_pos, int *lorenzoPred,
    size_t& cmpSize, size_t *prefix_length
){
    int lorenzo_pred, max_lorenzo = 0;
    int temp_fixed_rate, temp_outlier;
    for(int j=0; j<block_size; j++){
        lorenzo_pred = lorenzoPred[j];
        signFlag[j] = (lorenzo_pred < 0);
        absLorenzo[j] = abs(lorenzo_pred);
        if(j != 0)
            max_lorenzo = max_lorenzo > absLorenzo[j] ? max_lorenzo : absLorenzo[j];
        else
            temp_outlier = lorenzo_pred;
    }
    temp_fixed_rate = max_lorenzo == 0 ? 0 : INT_BITS - __builtin_clz(max_lorenzo);
    fixedRate[curr_block] = (unsigned char)temp_fixed_rate;
    if(temp_outlier < 0){
        temp_outlier = 0x80000000 | abs(temp_outlier);
    }
    for(int k=3; k>=0; k--){
        *(cmpData_pos++) = (temp_outlier >> (8 * k)) & 0xff;
    }
    size_t length = 4;
    if(temp_fixed_rate){
        unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
        cmpData_pos += signbyteLength;
        length += signbyteLength;
        unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absLorenzo, block_size, cmpData_pos, temp_fixed_rate);
        cmpData_pos += savedbitsbyteLength;
        length += savedbitsbyteLength;
    }
    *prefix_length += length;
    cmpSize += length;
    new_offsets[curr_block] = *prefix_length;
}

inline void integerize_lorenzo(
    int bias, int center, int index,
    int& residual, int * lorenzo
){
    // residual += center;
    // lorenzo[index] = (residual + bias) >> 2;
    // residual -= (lorenzo[index] << 2);
    auto predict = residual + center + bias;
    lorenzo[index] = predict >> 2;
    residual = (predict & 0x3) - bias;
}

inline void process_internal_lorenzo_rowwise_1d_block(
    int blockSize, int bias, int& residual,
    int *fixedRate_pos, int *update_lorenzo,
    int *prevRow, int *currRow, int *nextRow
){
    unsigned char prev = (fixedRate_pos[0] > 0);
    unsigned char curr = (fixedRate_pos[1] > 0);
    unsigned char next = (fixedRate_pos[2] > 0);

    unsigned char type = prev | (curr << 1) | (next << 2);
    int tmp;
    switch (type)
    {
    case 0:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, 0, j, residual, update_lorenzo);
        }
        break;
    case 1:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, prevRow[j], j, residual, update_lorenzo);
        }
        break;
    case 2:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, currRow[j-1] + currRow[j+1], j, residual, update_lorenzo);
        }
        break;
    case 3:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, prevRow[j] + currRow[j-1] + currRow[j+1], j, residual, update_lorenzo);
        }
        break;
    case 4:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, nextRow[j], j, residual, update_lorenzo);
        }
        break;
    case 5:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, prevRow[j] + nextRow[j], j, residual, update_lorenzo);
        }
        break;
    case 6:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, nextRow[j] + currRow[j-1] + currRow[j+1], j, residual, update_lorenzo);
        }
        break;
    case 7:
        for(int j=2; j<blockSize-1; j++){
            integerize_lorenzo(bias, prevRow[j] + nextRow[j] + currRow[j-1] + currRow[j+1], j, residual, update_lorenzo);
        }
        break;
    } 
}

void update_lorenzoPred_rowwise_1d_block(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int *absLorenzo, unsigned char *signFlag, int *update_lorenzo,
    int *prevRow_lorenzo, int *currRow_lorenzo, int *nextRow_lorenzo,
    size_t dim1, size_t dim2, int blockSize, size_t& cmpSize,
    int q_S, int q_W, int q_B,
    int current, int next, int iter
){
    int x, j;
    int center, residual;
    int bias = (iter & 1) + 1;
    size_t prefix_length = 0;
    int block_num = dim1;
    int currRow_second, currRow_last;
    int nextRow_second, nextRow_last;
    int * temp = NULL;
    unsigned char * nextRow_cmpData = NULL;
    unsigned char * cmpData_pos = cmpData[current] + block_num;
    unsigned char * cmpData_pos_update = cmpData[next] + block_num;
    // first row
    {
        residual = 0;
        x = 0;
        decompressToLorenzo_rowwise_1d_block(x, blockSize, absLorenzo, signFlag, fixedRate[current], cmpData_pos,
                                            currRow_lorenzo, &currRow_second, &currRow_last);
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decompressToLorenzo_rowwise_1d_block(x+1, blockSize, absLorenzo, signFlag, fixedRate[current], nextRow_cmpData,
                                            nextRow_lorenzo, &nextRow_second, &nextRow_last);
        j = 0;
        center = q_W + currRow_second + q_S + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        for(j=2; j<blockSize-1; j++){
            center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + nextRow_lorenzo[j];
            integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        }
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        compressFromLorenzo_rowwise_1d_block(x, blockSize, absLorenzo, signFlag, fixedRate[next], offsets[next],
                                            cmpData_pos_update, update_lorenzo, cmpSize, &prefix_length);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // row 1 ~ (dim1-2)
    for(x=1; x<dim1-1; x++){
        residual = 0;
        temp = prevRow_lorenzo;
        prevRow_lorenzo = currRow_lorenzo;
        currRow_lorenzo = nextRow_lorenzo;
        nextRow_lorenzo = temp;
        currRow_second = nextRow_second;
        currRow_last = nextRow_last;
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decompressToLorenzo_rowwise_1d_block(x+1, blockSize, absLorenzo, signFlag, fixedRate[current], nextRow_cmpData,
                                            nextRow_lorenzo, &nextRow_second, &nextRow_last);
        j = 0;
        center = q_W + currRow_second + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        // for(j=2; j<blockSize-1; j++){
        //     center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        //     integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        // }
        process_internal_lorenzo_rowwise_1d_block(blockSize, bias, residual, fixedRate[current]+x-1, update_lorenzo, prevRow_lorenzo, currRow_lorenzo, nextRow_lorenzo);
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        compressFromLorenzo_rowwise_1d_block(x, blockSize, absLorenzo, signFlag, fixedRate[next], offsets[next],
                                            cmpData_pos_update+offsets[next][x-1], update_lorenzo, cmpSize, &prefix_length);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // last row
    {
        residual = 0;
        x = dim1 - 1;
        prevRow_lorenzo = currRow_lorenzo;
        currRow_lorenzo = nextRow_lorenzo;
        currRow_second = nextRow_second;
        currRow_last = nextRow_last;
        j = 0;
        center = q_W + currRow_second + prevRow_lorenzo[j] + q_B;
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + prevRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        for(j=2; j<blockSize-1; j++){
            center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + prevRow_lorenzo[j];
            integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        }
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + prevRow_lorenzo[j];
        integerize_lorenzo(bias, center, j, residual, update_lorenzo);
        compressFromLorenzo_rowwise_1d_block(x, blockSize, absLorenzo, signFlag, fixedRate[next], offsets[next],
                                            cmpData_pos_update+offsets[next][x-1], update_lorenzo, cmpSize, &prefix_length);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
}

void SZp_heatdis_kernel_lorenzo_rowwise_1d_block(
    unsigned char **cmpData, int **offsets, int **fixedRate,
    unsigned int * absLorenzo, unsigned char *signFlag,
    size_t dim1, size_t dim2, int blockSize,
    double errorBound, size_t *cmpSize, int max_iter
){
    int block_num = dim1;
    int curr_block, temp_fixed_rate;
    int cmp_block_sign_length = (blockSize + 7) / 8;
    int current = 0, next = 1;
    unsigned int temp_outlier;
    unsigned char * temp_sign_flag = signFlag;
    int temp_ofs_outlier;
    size_t prefix_length = 0;
    for(int x=0; x<dim1; x++){
        curr_block = x;
        temp_fixed_rate = (int)cmpData[current][curr_block];
        fixedRate[current][curr_block] = temp_fixed_rate;
        size_t savedbitsbytelength = compute_encoding_byteLength(blockSize, temp_fixed_rate);
        prefix_length += 4;
        if(temp_fixed_rate) 
            prefix_length += cmp_block_sign_length + savedbitsbytelength;
        offsets[current][curr_block] = prefix_length;
    }
    const int q_S = static_cast<int>(std::floor((SRC_TEMP + errorBound) / (2 * errorBound)));
    const int q_W = static_cast<int>(std::floor((WALL_TEMP + errorBound) / (2 * errorBound)));
    const int q_B = static_cast<int>(std::floor((BACK_TEMP + errorBound) / (2 * errorBound)));
    int * prevRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * currRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * nextRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * update_lorenzo = (int *)calloc(blockSize, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        update_lorenzoPred_rowwise_1d_block(cmpData, offsets, fixedRate, absLorenzo, signFlag,
                                            update_lorenzo, prevRow_lorenzo, currRow_lorenzo, nextRow_lorenzo,
                                            dim1, dim2, blockSize, compressed_size, q_S, q_W, q_B, current, next, iter);
        current = next;
        next = 1 - current;
    }
    *cmpSize = compressed_size;
    free(prevRow_lorenzo);
    free(currRow_lorenzo);
    free(nextRow_lorenzo);
    free(update_lorenzo);
}

#endif