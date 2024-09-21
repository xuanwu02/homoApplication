#ifndef _STENCIL_HEATDIS_1B_HPP
#define _STENCIL_HEATDIS_1B_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "ompSZp_typemanager.hpp"

const double SRC_TEMP = 100.0;
const double WALL_TEMP = 0.0;
const double BACK_TEMP = 0.0;

void SZp_compress_kernel_1D(float *oriData, unsigned char *cmpData, size_t dim1, size_t dim2,
                          unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
                          double errorBound, int blockSize, size_t *cmpSize)
{
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int tailSize = dim2 - blockSize * (row_block_num - 1);
    int block_num = row_block_num * dim1;
    const double recip_precision = 0.5f / errorBound;
    unsigned char * cmpData_pos = cmpData + block_num;
    for(int x=0; x<dim1; x++){
        for(int y=0; y<row_block_num; y++){
            int curr_block = x * row_block_num + y;
            int curr_size = (y < row_block_num - 1) ? blockSize : tailSize;  
            int offset = x * dim2;
            int prefix = offset + y * blockSize;
            double data_recip;
            int s;
            int curr_quant, lorenzo_pred, prev_quant = 0;
            int max_lorenzo = 0;
            int temp_fixed_rate;
            int temp_outlier;
            int index;
            for(int j=0; j<curr_size; j++){
                index = prefix + j;
                data_recip = oriData[index] * recip_precision;
                s = data_recip >= -0.5f ? 0 : 1;
                curr_quant = (int)(data_recip + 0.5f) - s;
                lorenzo_pred = curr_quant - prev_quant;
                prev_quant = curr_quant;
                signFlag[j] = (lorenzo_pred < 0);
                absLorenzo[j] = abs(lorenzo_pred);
                if(j != 0)
                    max_lorenzo = max_lorenzo > absLorenzo[j] ? max_lorenzo : absLorenzo[j];
                else
                    temp_outlier = lorenzo_pred;
            }
            temp_fixed_rate = max_lorenzo == 0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_lorenzo);
            cmpData[curr_block] = (unsigned char)temp_fixed_rate;
            if(temp_outlier < 0) temp_outlier = 0x80000000 | abs(temp_outlier);
            for(int k=3; k>=0; k--){
                *(cmpData_pos++) = (temp_outlier >> (8 * k)) & 0xff;
            }
            if(temp_fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, curr_size, cmpData_pos);
                cmpData_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absLorenzo, curr_size, cmpData_pos, temp_fixed_rate);
                cmpData_pos += savedbitsbyteLength;
            }
        }
    }
    *cmpSize = cmpData_pos - cmpData;
}

void SZp_decompress_kernel_1D(float *decData, unsigned char *cmpData, size_t dim1, size_t dim2,
                            unsigned int *absLorenzo, unsigned char *signFlag, int *fixedRate,
                            double errorBound, int blockSize)
{
    int row_block_num = (dim2 - 1) / blockSize + 1;
    int tailSize = dim2 - blockSize * (row_block_num - 1);
    int block_num = row_block_num * dim1;
    unsigned char * cmpData_pos = cmpData + block_num;
    size_t cmp_block_sign_length;
    for(int x=0; x<dim1; x++){
        for(int y=0; y<row_block_num; y++){
            int curr_block = x * row_block_num + y;
            int curr_size = (y < row_block_num - 1) ? blockSize : tailSize;  
            int offset = x * dim2;
            int prefix = offset + y * blockSize;
            int temp_fixed_rate = (int)cmpData[curr_block];
            int temp_outlier = (0xff000000 & (*cmpData_pos << 24)) |
                                (0x00ff0000 & (*(cmpData_pos+1) << 16)) |
                                (0x0000ff00 & (*(cmpData_pos+2) << 8)) |
                                (0x000000ff & *(cmpData_pos+3));
            cmpData_pos += 4;
            cmp_block_sign_length = (curr_size + 7) / 8;
            int index;
            if(temp_fixed_rate){
                convertByteArray2IntArray_fast_1b_args(curr_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, curr_size, absLorenzo, temp_fixed_rate);
                cmpData_pos += savedbitsbytelength;
                absLorenzo[0] = temp_outlier & 0x7fffffff;
                int curr_quant, lorenzo_pred, prev_quant = 0;
                for(int j=0; j<curr_size; j++){
                    index = prefix + j;
                    if(signFlag[j])
                        lorenzo_pred = absLorenzo[j] * -1;
                    else
                        lorenzo_pred = absLorenzo[j];
                    curr_quant = lorenzo_pred + prev_quant;
                    decData[index] = curr_quant * errorBound * 2;
                    prev_quant = curr_quant;
                }
            }
            else if(!temp_fixed_rate){
                int first_sign = (temp_outlier & 0x80000000) >> 31;
                int tar_quant = 0;
                if(first_sign)
                    tar_quant = (temp_outlier & 0x7fffffff) * -1;
                else
                    tar_quant = (temp_outlier & 0x7fffffff);
                if(!temp_outlier) tar_quant = 0;
                for(int j=0; j<curr_size; j++){
                    index = prefix + j;
                    decData[index] = tar_quant * errorBound * 2;
                }
            }
        }
    }
}

size_t compute_byteLength(size_t intArrayLength, int bit_count)
{
    unsigned int byte_count = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    size_t byteLength = byte_count * intArrayLength + (remainder_bit * intArrayLength - 1) / 8 + 1;
    if(!bit_count) byteLength = 0;
    return byteLength;
}

void decomressToQuant_single_block(int curr_block, int block_size,
                                   int *fixedRate, unsigned char *cmpData_pos,
                                   unsigned char *signFlag, unsigned int *absLorenzo, int *quantInds)
{
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
            if(signFlag[j])
                lorenzo_pred = absLorenzo[j] * -1;
            else
                lorenzo_pred = absLorenzo[j];
            curr_quant = lorenzo_pred + prev_quant;
            quantInds[j] = curr_quant;
            prev_quant = curr_quant; 
        }
    }
    else if(!temp_fixed_rate){
        int first_sign = (temp_outlier & 0x80000000) >> 31;
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

void compressFromQuant_single_block(int curr_block, int block_size, size_t& cmpSize, size_t *prefix_length,
                                    int *new_offsets, int *fixedRate, unsigned char *cmpData_pos,
                                    unsigned char *signFlag, unsigned int *absLorenzo, int *quantInds)
{
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
    temp_fixed_rate = max_lorenzo == 0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_lorenzo);
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

void update_quantInds(unsigned char **cmpData, int **offsets, int **fixedRate,
                    unsigned int *absLorenzo, unsigned char *signFlag, int *update_quant,
                    int *prevRow_quant, int *currRow_quant, int *nextRow_quant,
                    size_t dim1, size_t dim2, int q_S, int q_W, int q_B, int blockSize,
                    int current, int next, int iter, size_t& cmpSize)
{
    int x, j;
    double center;
    size_t prefix_length = 0;
    int block_num = dim1;
    int * temp = NULL;
    unsigned char * nextRow_cmpData = NULL;
    unsigned char * cmpData_pos = cmpData[current] + block_num;
    unsigned char * cmpData_pos_update = cmpData[next] + block_num;
    // first row
    {
        x = 0;
        decomressToQuant_single_block(x, blockSize, fixedRate[current], cmpData_pos, signFlag, absLorenzo, currRow_quant);
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decomressToQuant_single_block(x + 1, blockSize, fixedRate[current], nextRow_cmpData, signFlag, absLorenzo, nextRow_quant);
        j = 0;
        center = 0.25 * (q_W + currRow_quant[j+1] + q_S + nextRow_quant[j]);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        for(j=1; j<blockSize-1; j++){
            center = 0.25 * (currRow_quant[j-1] + currRow_quant[j+1] + q_S + nextRow_quant[j]);
            update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        }
        j = blockSize - 1;
        center = 0.25 * (currRow_quant[j-1] + q_W + q_S + nextRow_quant[j]);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        compressFromQuant_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                        cmpData_pos_update, signFlag, absLorenzo, update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // row 1 ~ (dim1-2)
    for(x=1; x<dim1-1; x++){
        temp = prevRow_quant;
        prevRow_quant = currRow_quant;
        currRow_quant = nextRow_quant;
        nextRow_quant = temp;
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decomressToQuant_single_block(x + 1, blockSize, fixedRate[current], nextRow_cmpData, signFlag, absLorenzo, nextRow_quant);
        j = 0;
        center = 0.25 * (q_W + currRow_quant[j+1] + prevRow_quant[j] + nextRow_quant[j]);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        for(j=1; j<blockSize-1; j++){
            center = 0.25 * (currRow_quant[j-1] + currRow_quant[j+1] + prevRow_quant[j] + nextRow_quant[j]);
            update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        }
        j = blockSize - 1;
        center = 0.25 * (currRow_quant[j-1] + q_W + prevRow_quant[j] + nextRow_quant[j]);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        compressFromQuant_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                        cmpData_pos_update + offsets[next][x-1], signFlag, absLorenzo, update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // last row
    {
        x = dim1 - 1;
        prevRow_quant = currRow_quant;
        currRow_quant = nextRow_quant;
        j = 0;
        center = 0.25 * (q_W + currRow_quant[j+1] + prevRow_quant[j] + q_B);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        for(j=1; j<blockSize-1; j++){
            center = 0.25 * (currRow_quant[j-1] + currRow_quant[j+1] + prevRow_quant[j] + q_B);
            update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        }
        j = blockSize - 1;
        center = 0.25 * (currRow_quant[j-1] + q_W + prevRow_quant[j] + q_B);
        update_quant[j] = center + (center >= -0.5 ? 0.5 : -0.5);
        compressFromQuant_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                        cmpData_pos_update + offsets[next][x-1], signFlag, absLorenzo, update_quant);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
}

void SZp_heatdis_kernel_decomressToQuant(unsigned char **cmpData, int **& offsets, int **& fixedRate,
                                        unsigned int *absLorenzo, unsigned char *signFlag,
                                        size_t dim1, size_t dim2, double errorBound, int blockSize,
                                        size_t *cmpSize, int max_iter)
{
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
        size_t savedbitsbytelength = compute_byteLength(blockSize, temp_fixed_rate);
        prefix_length += 4;
        if(temp_fixed_rate) 
            prefix_length += cmp_block_sign_length + savedbitsbytelength;
        offsets[current][curr_block] = prefix_length;
    }
    const double recip_precision = 0.5f / errorBound;
    const int q_S = (int)(SRC_TEMP * recip_precision + 0.5f);
    const int q_W = (int)(WALL_TEMP * recip_precision + 0.5f);
    const int q_B = (int)(BACK_TEMP * recip_precision + 0.5f);
    int * prevRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * currRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * nextRow_quant = (int *)calloc(blockSize, sizeof(int));
    int * update_quant = (int *)calloc(blockSize, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        update_quantInds(cmpData, offsets, fixedRate, absLorenzo, signFlag,
                        update_quant, prevRow_quant, currRow_quant, nextRow_quant,
                        dim1, dim2, q_S, q_W, q_B, blockSize, current, next, iter, compressed_size);
        current = next;
        next = 1 - current;
    }
    *cmpSize = compressed_size;
    free(prevRow_quant);
    free(currRow_quant);
    free(nextRow_quant);
    free(update_quant);
}

void decomressToLorenzo_single_block(int curr_block, int block_size,
                                     int *offsets, int *fixedRate, unsigned char *cmpData_pos,
                                     unsigned char *signFlag, unsigned int *absLorenzo,
                                     int *second_quant, int *last_quant, int *lorenzoPred)
{
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
            if(signFlag[j])
                lorenzo = absLorenzo[j] * -1;
            else
                lorenzo = absLorenzo[j];
            *last_quant += lorenzo;
            lorenzoPred[j] = lorenzo;
        }
        *second_quant = lorenzoPred[0] + lorenzoPred[1];
    }
    else if(!temp_fixed_rate){
        int first_sign = (temp_outlier & 0x80000000) >> 31;
        int tar_quant = 0;
        if(first_sign)
            tar_quant = (temp_outlier & 0x7fffffff) * -1;
        else
            tar_quant = (temp_outlier & 0x7fffffff);
        lorenzoPred[0] = tar_quant;
        for(int j=1; j<block_size; j++){
            lorenzoPred[j] = 0;
        }
        *second_quant = tar_quant;
        *last_quant = tar_quant;
    }
}

void compressFromLorenzo_single_block(int curr_block, int block_size, size_t& cmpSize, size_t *prefix_length,
                                      int *new_offsets, int *fixedRate, unsigned char *cmpData_pos,
                                      unsigned char *signFlag, unsigned int *absLorenzo, int *lorenzoPred)
{
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
    temp_fixed_rate = max_lorenzo == 0 ? 0 : sizeof(int) * 8 - __builtin_clz(max_lorenzo);
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

inline void integerize(int bias, int center, int& residual, int& lorenzo)
{
    residual += center;
    lorenzo = (residual + bias) >> 2;
    residual -= (lorenzo << 2);
}

void update_lorenzoPred(unsigned char **cmpData, int **offsets, int **fixedRate,
                        unsigned int *absLorenzo, unsigned char *signFlag, int *update_lorenzo,
                        int *prevRow_lorenzo, int *currRow_lorenzo, int *nextRow_lorenzo,
                        size_t dim1, size_t dim2, int q_S, int q_W, int q_B, int blockSize,
                        int current, int next, int iter, size_t& cmpSize)
{
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
        decomressToLorenzo_single_block(x, blockSize, offsets[current], fixedRate[current], cmpData_pos,
                                        signFlag, absLorenzo, &currRow_second, &currRow_last, currRow_lorenzo);
        nextRow_cmpData = cmpData_pos + offsets[current][x];
        decomressToLorenzo_single_block(x + 1, blockSize, offsets[current], fixedRate[current], nextRow_cmpData,
                                        signFlag, absLorenzo, &nextRow_second, &nextRow_last, nextRow_lorenzo);
        j = 0;
        center = q_W + currRow_second + q_S + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        for(j=2; j<blockSize-1; j++){
            center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + nextRow_lorenzo[j];
            integerize(bias, center, residual, update_lorenzo[j]);
        }
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        compressFromLorenzo_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                         cmpData_pos_update, signFlag, absLorenzo, update_lorenzo);
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
        decomressToLorenzo_single_block(x + 1, blockSize, offsets[current], fixedRate[current], nextRow_cmpData,
                                        signFlag, absLorenzo, &nextRow_second, &nextRow_last, nextRow_lorenzo);
        j = 0;
        center = q_W + currRow_second + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        for(j=2; j<blockSize-1; j++){
            center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + prevRow_lorenzo[j] + nextRow_lorenzo[j];
            integerize(bias, center, residual, update_lorenzo[j]);
        }
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + prevRow_lorenzo[j] + nextRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        compressFromLorenzo_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                         cmpData_pos_update + offsets[next][x-1], signFlag, absLorenzo, update_lorenzo);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
    // last row
    {
        x = dim1 - 1;
        prevRow_lorenzo = currRow_lorenzo;
        currRow_lorenzo = nextRow_lorenzo;
        currRow_second = nextRow_second;
        currRow_last = nextRow_last;
        j = 0;
        center = q_W + currRow_second + prevRow_lorenzo[j] + q_B;
        integerize(bias, center, residual, update_lorenzo[j]);
        j = 1;
        center = currRow_lorenzo[j-1] - q_W + currRow_lorenzo[j+1] + prevRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        for(j=2; j<blockSize-1; j++){
            center = currRow_lorenzo[j-1] + currRow_lorenzo[j+1] + prevRow_lorenzo[j];
            integerize(bias, center, residual, update_lorenzo[j]);
        }
        j = blockSize - 1;
        center = currRow_lorenzo[j-1] + q_W - currRow_last + prevRow_lorenzo[j];
        integerize(bias, center, residual, update_lorenzo[j]);
        compressFromLorenzo_single_block(x, blockSize, cmpSize, &prefix_length, offsets[next], fixedRate[next],
                                         cmpData_pos_update + offsets[next][x-1], signFlag, absLorenzo, update_lorenzo);
        cmpData[next][x] = (unsigned char)fixedRate[next][x];
    }
}

void SZp_heatdis_kernel_decomressToLorenzo(unsigned char **cmpData, int **offsets, int **fixedRate,
                                            unsigned int * absLorenzo, unsigned char *signFlag,
                                            size_t dim1, size_t dim2, double errorBound, int blockSize,
                                            size_t *cmpSize, int max_iter)
{
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
        size_t savedbitsbytelength = compute_byteLength(blockSize, temp_fixed_rate);
        prefix_length += 4;
        if(temp_fixed_rate) 
            prefix_length += cmp_block_sign_length + savedbitsbytelength;
        offsets[current][curr_block] = prefix_length;
    }
    const double recip_precision = 0.5f / errorBound;
    const int q_S = (int)(SRC_TEMP * recip_precision + 0.5f);
    const int q_W = (int)(WALL_TEMP * recip_precision + 0.5f);
    const int q_B = (int)(BACK_TEMP * recip_precision + 0.5f);
    int * prevRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * currRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * nextRow_lorenzo = (int *)calloc(blockSize, sizeof(int));
    int * update_lorenzo = (int *)calloc(blockSize, sizeof(int));
    size_t compressed_size;
    for(int iter=0; iter<max_iter; iter++){
        compressed_size = block_num;
        update_lorenzoPred(cmpData, offsets, fixedRate, absLorenzo, signFlag,
                            update_lorenzo, prevRow_lorenzo, currRow_lorenzo, nextRow_lorenzo,
                            dim1, dim2, q_S, q_W, q_B, blockSize, current, next, iter, compressed_size);
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