#ifndef _SZX_MEAN_PREDICTOR_2D_HPP
#define _SZX_MEAN_PREDICTOR_2D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include "typemanager.hpp"
#include "SZ_app_utils.hpp"
#include "utils.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZx_compress(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * block_quant_inds = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
    size_t prefix = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate, max_err = 0;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            int * block_buffer_pos = block_quant_inds;
            int mean_quant = compute_block_mean_quant(size_x, size_y, size.offset_0, y_data_pos, block_buffer_pos, inver_eb);
            for(int i=0; i<block_size; i++){
                int err = *block_buffer_pos++ - mean_quant;
                int abs_err = abs(err);
                *sign_pos++ = (err < 0);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            memcpy(qmean_pos, &mean_quant, sizeof(int));
            qmean_pos += 4;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                prefix += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
                prefix += savedbitsbyteLength;
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(block_quant_inds);
}

template <class T>
void SZx_decompress(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    double twice_eb = 2 * errorBound;
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean_quant = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            T * curr_data_pos = y_data_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr = 0 - absPredError[index];
                        // else curr = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr = (absPredError[index] ^ s) - s;
                        index++;
                        *curr_data_pos++ = (curr + mean_quant) * twice_eb;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        *curr_data_pos++ = mean_quant * twice_eb;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_meta(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    free(blocks_mean_quant);
}

void SZx_decompress_postPred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * curr_data_pos = y_data_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr = 0 - absPredError[index];
                        // else curr = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr = (absPredError[index] ^ s) - s;
                        index++;
                        *curr_data_pos++ = curr;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        *curr_data_pos++ = 0;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_prePred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean_quant = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            int * curr_data_pos = y_data_pos;
            int curr;
            int index;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr = 0 - absPredError[index];
                        // else curr = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr = (absPredError[index] ^ s) - s;
                        index++;
                        *curr_data_pos++ = curr + mean_quant;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        *curr_data_pos++ = mean_quant;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time = get_elapsed_time(start2, end2);
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

double SZx_mean_meta(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int64_t sum = 0;
    int block_mean;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            memcpy(&block_mean, qmean_pos, sizeof(int));
            qmean_pos += 4;
            sum += block_mean * block_size;
        }
    }
    double mean = sum * 2 * errorBound / size.nbEle;
    return mean;
}

double SZx_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int64_t quant_sum = 0;
    int block_ind = 0;
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean_quant = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            int err;
            quant_sum += mean_quant * block_size;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(int i=0; i<block_size; i++){
                    if(signFlag[i]) err = 0 - absPredError[i];
                    else err = absPredError[i];
                    quant_sum += err;
                }
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = 2 * errorBound * (double)quant_sum / size.nbEle;
    return mean;
}

double SZx_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int64_t quant_sum = 0;
    int block_ind = 0;
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean_quant = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            int curr;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(int i=0; i<block_size; i++){
                    if(signFlag[i]) curr = 0 - absPredError[i];
                    else curr = absPredError[i];
                    curr += mean_quant;
                    quant_sum += curr;
                }
            }else{
                quant_sum += mean_quant * block_size;
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZx_mean_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZx_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZx_mean_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            mean = SZx_mean_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_mean_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::meta:{
            mean = SZx_mean_meta(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_region_mean_meta(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * size.Bsize * size.Bsize;
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y;
    int block_ind = 0;
    int size_x, size_y, block_size;
    int mean_quant;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        block_ind = x * size.block_dim2 + lo2;
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            mean_quant = blocks_mean_quant[block_ind++];
            quant_sum += mean_quant * block_size;
        }
    }
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

double SZx_region_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * size.Bsize * size.Bsize;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    const unsigned char * rate_start_pos = cmpData;
    unsigned char * encode_start_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    std::vector<size_t> prefix(hi1 - lo1, 0);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y, i;
    size_t byteLengthPrefix = 0;
    int block_ind = 0;
    int size_x, size_y, block_size, fixed_rate;
    int mean_quant, curr;
    for(x=0; x<lo1; x++){
        for(y=0; y<size.block_dim2; y++){
            size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    i = 0;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=0; y<lo2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
        prefix[i++] = byteLengthPrefix;
        for(y=lo2; y<size.block_dim2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        block_ind = x * size.block_dim2 + lo2;
        unsigned char * encode_pos = encode_start_pos + prefix[x - lo1];
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            mean_quant = blocks_mean_quant[block_ind];
            fixed_rate = (int)rate_start_pos[block_ind++];
            quant_sum += mean_quant * block_size;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(i=0; i<block_size; i++){
                    if(signFlag[i]) curr = 0 - absPredError[i];
                    else curr = absPredError[i];
                    quant_sum += curr;
                }
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

double SZx_region_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * size.Bsize * size.Bsize;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    const unsigned char * rate_start_pos = cmpData;
    unsigned char * encode_start_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    std::vector<size_t> prefix(hi1 - lo1, 0);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y, i;
    size_t byteLengthPrefix = 0;
    int block_ind = 0;
    int size_x, size_y, block_size, fixed_rate;
    int mean_quant, curr;
    for(x=0; x<lo1; x++){
        for(y=0; y<size.block_dim2; y++){
            size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    i = 0;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=0; y<lo2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
        prefix[i++] = byteLengthPrefix;
        for(y=lo2; y<size.block_dim2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        block_ind = x * size.block_dim2 + lo2;
        unsigned char * encode_pos = encode_start_pos + prefix[x - lo1];
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            mean_quant = blocks_mean_quant[block_ind];
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(i=0; i<block_size; i++){
                    if(signFlag[i]) curr = 0 - absPredError[i];
                    else curr = absPredError[i];
                    curr += mean_quant;
                    quant_sum += curr;
                }
            }else{
                quant_sum += mean_quant * block_size;
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

template <class T>
double SZx_region_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    double ratio, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            SZx_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            mean = compute_region_mean(dim1, dim2, blockSideLength, ratio, decData);
            break;
        }
        case decmpState::prePred:{
            mean = SZx_region_mean_prePred(cmpData, dim1, dim2, ratio, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_region_mean_postPred(cmpData, dim1, dim2, ratio, blockSideLength, errorBound);            
            break;
        }
        case decmpState::meta:{
            mean = SZx_region_mean_meta(cmpData, dim1, dim2, ratio, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_stddev_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_2d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    uint64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            uint64_t block_size = size_x * size_y;
            int block_mean = blocks_mean_quant[block_ind];
            int mean_err = block_mean - global_mean;
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<block_size; i++){
                    // if(signFlag[index]) curr = 0 - absPredError[index];
                    // else curr = absPredError[index];
                    int s = -(int)signFlag[index];
                    curr = (absPredError[index] ^ s) - s;
                    index++;
                    int64_t d = static_cast<int64_t>(curr + mean_err);
                    uint64_t d2 = d * d;
                    squared_sum += d2;
                }
            }else{
                int64_t d = static_cast<int64_t>(mean_err);
                uint64_t d2 = d * d;
                squared_sum += d2 * block_size;
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double std = (2 * errorBound) * sqrt((double)squared_sum / (size.nbEle - 1));
    return std;
}

double SZx_stddev_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    extract_block_mean(qmean_pos, blocks_mean_quant, size.num_blocks);
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int block_mean = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<block_size; i++){
                    // if(signFlag[index]) curr = 0 - absPredError[index];
                    // else curr = absPredError[index];
                    int s = -(int)signFlag[index];
                    curr = (absPredError[index] ^ s) - s;
                    index++;
                    int64_t d = static_cast<int64_t>(curr + block_mean);
                    uint64_t d2 = d * d;
                    quant_sum += d;
                    squared_quant_sum += d2;
                }
            }else{
                int64_t d = static_cast<int64_t>(block_mean);
                uint64_t d2 = d * d;
                quant_sum += d * block_size;
                squared_quant_sum += d2 * block_size;
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double std = (2 * errorBound) * sqrt(((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1));
    return std;
}

template <class T>
double SZx_stddev_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double std = 0;
    for(size_t i=0; i<nbEle; i++) std += (decData[i] - mean) * (decData[i] - mean);
    std /= (nbEle - 1);
    return sqrt(std);
}

template <class T>
double SZx_stddev(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double std;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            std = SZx_stddev_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            std = SZx_stddev_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            std = SZx_stddev_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return std;
}

// TODO
double SZx_region_dxdy_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    double ratio, int blockSideLength, double errorBound,
    int *buffer_data_pos, size_t offset_0
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * size.Bsize * size.Bsize;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    const unsigned char * rate_start_pos = cmpData;
    unsigned char * encode_start_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    std::vector<size_t> prefix(hi1 - lo1, 0);
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y, i;
    size_t byteLengthPrefix = 0;
    int block_ind = 0;
    int size_x, size_y, block_size, fixed_rate;
    int mean_quant, curr;
    for(x=0; x<lo1; x++){
        for(y=0; y<size.block_dim2; y++){
            size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    i = 0;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=0; y<lo2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
        prefix[i++] = byteLengthPrefix;
        for(y=lo2; y<size.block_dim2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
            }
        }
    }
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        block_ind = x * size.block_dim2 + lo2;
        unsigned char * encode_pos = encode_start_pos + prefix[x - lo1];
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_size = size_x * size_y;
            mean_quant = blocks_mean_quant[block_ind];
            fixed_rate = (int)rate_start_pos[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(i=0; i<block_size; i++){
                    if(signFlag[i]) curr = 0 - absPredError[i];
                    else curr = absPredError[i];
                    curr += mean_quant;
                    quant_sum += curr;
                }
            }else{
                quant_sum += mean_quant * block_size;
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

inline void recoverBlockSlice2PrePred(
    size_t x, DSize_2d size, CmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t offset_0
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    int block_ind_offset = x * size.block_dim2;
    unsigned char * cmpData = cmpkit_set->compressed;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_ind = block_ind_offset + y;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind];
        int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = mean_quant;
                }
                curr_buffer_pos += offset_0;
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            int curr;
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    // if(cmpkit_set->signFlag[index]) curr = 0 - cmpkit_set->absPredError[index];
                    // else curr = cmpkit_set->absPredError[index];
                    int s = -(int)cmpkit_set->signFlag[index];
                    curr = (cmpkit_set->absPredError[index] ^ s) - s;
                    index++;
                    curr_buffer_pos[j] = curr + mean_quant;
                }
                curr_buffer_pos += offset_0;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockSlice2PostPred(
    size_t x, DSize_2d size, CmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t offset_0
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    int block_ind = x * size.block_dim2;
    unsigned char * cmpData = cmpkit_set->compressed;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind++];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = 0;
                }
                curr_buffer_pos += offset_0;
            }        
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    // if(cmpkit_set->signFlag[index]) curr_buffer_pos[j] = 0 - cmpkit_set->absPredError[index];
                    // else curr_buffer_pos[j] = cmpkit_set->absPredError[index];
                    int s = -(int)cmpkit_set->signFlag[index];
                    curr_buffer_pos[j] = (cmpkit_set->absPredError[index] ^ s) - s;
                    index++;
                }
                curr_buffer_pos += offset_0;
            }        
        }
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockSlicePrePred(
    size_t x, DSize_2d size, AppBufferSet_2d *buffer_set,
    CmpBufferSet *cmpkit_set, T *dx_start_pos, T *dy_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            dx_pos[index] = (next_row[j] - prev_row[j]) * errorBound;
            dy_pos[index] = (curr_row[j+1] - curr_row[j-1]) * errorBound;
            index++;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockSlicePostPred(
    size_t x, DSize_2d size, AppBufferSet_2d *buffer_set,
    T *row_diffs, T *rowpair_diffs, T *dx_start_pos, T *dy_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * buffer_start_pos = buffer_set->currSlice_data_pos;
    const T * dy_blockmean_diff = row_diffs + x * (size.block_dim2 + 1);
    const T * dx_top_blockmean_diff = rowpair_diffs + x * size.block_dim2;
    const T * dx_bott_blockmean_diff = rowpair_diffs + (x + 1) * size.block_dim2;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        T * dx_pos = dx_start_pos + y * size.Bsize;
        T * dy_pos = dy_start_pos + y * size.Bsize;
        const int * curr_row = buffer_start_pos;
        for(int i=0; i<size_x; i++){
            const int * prev_row = curr_row - buffer_set->offset_0;
            const int * next_row = curr_row + buffer_set->offset_0;
            for(int j=0; j<size_y; j++){
                dx_pos[j] = (next_row[j] - prev_row[j]) * errorBound;
                dy_pos[j] = (curr_row[j+1] - curr_row[j-1]) * errorBound;
            }
            dy_pos[0] += dy_blockmean_diff[y];
            dy_pos[size_y - 1] += dy_blockmean_diff[y+1];
            if(i == 0){
                for(int j=0; j<size_y; j++) dx_pos[j] += dx_top_blockmean_diff[y];
            }
            if(i == size_x-1){
                for(int j=0; j<size_y; j++) dx_pos[j] += dx_bott_blockmean_diff[y];
            }
            dy_pos += size.offset_0;
            dx_pos += size.offset_0;
            curr_row += buffer_set->offset_0;
        }
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_2d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            dxdyProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
                dxdyProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdyProcessBlocksPostPred(
    DSize_2d& size,
    CmpBufferSet *cmpkit_set, 
    T * row_diffs, T * rowpair_diffs,
    AppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    dxdy_compute_block_mean_difference(size.block_dim1, size.block_dim2, errorBound, cmpkit_set->mean_quant_inds, row_diffs, rowpair_diffs);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            dxdyProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
                dxdyProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_dxdy(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    T * row_diffs = (T *)malloc(size.block_dim1*(size.block_dim2+1)*sizeof(T));
    T * rowpair_diffs = (T *)malloc(size.block_dim2*(size.block_dim1+1)*sizeof(T));
    int * Buffer_2d = (int *)malloc(buffer_dim1 * buffer_dim2 * 3 * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    AppBufferSet_2d * buffer_set = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, blocks_mean_quant);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdyProcessBlocksPostPred(size, cmpkit_set, row_diffs, rowpair_diffs, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_dxdy(dim1, dim2, decData, dx_pos, dy_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(row_diffs);
    free(rowpair_diffs);
    free(Buffer_2d);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize_2d size, AppBufferSet_2d *buffer_set,
    CmpBufferSet *cmpkit_set, T *result_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * laplacian_pos = result_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            laplacian_pos[index++] = (curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j]) * errorBound * 2;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlockSlicePostPred(
    size_t x, DSize_2d size, AppBufferSet_2d *buffer_set,
    T *row_diffs, T *rowpair_diffs, T *result_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const T * dy_blockmean_diff = row_diffs + x * (size.block_dim2 + 1);
    const T * dx_top_blockmean_diff = rowpair_diffs + x * size.block_dim2;
    const T * dx_bott_blockmean_diff = rowpair_diffs + (x + 1) * size.block_dim2;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * buffer_start_pos = buffer_set->currSlice_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        T * laplacian_pos = result_start_pos + y * size.Bsize;
        const int * curr_row = buffer_start_pos;
        for(int i=0; i<size_x; i++){
            const int * prev_row = curr_row - buffer_set->offset_0;
            const int * next_row = curr_row + buffer_set->offset_0;
            for(int j=0; j<size_y; j++){
                laplacian_pos[j] = (curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j]) * errorBound * 2;
            }
            laplacian_pos[0] -= dy_blockmean_diff[y];
            laplacian_pos[size_y - 1] += dy_blockmean_diff[y+1];
            if(i == 0){
                for(int j=0; j<size_y; j++) laplacian_pos[j] -= dx_top_blockmean_diff[y];
            }
            if(i == size_x-1){
                for(int j=0; j<size_y; j++) laplacian_pos[j] += dx_bott_blockmean_diff[y];
            }
            laplacian_pos += size.offset_0;
            curr_row += buffer_set->offset_0;
        }
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize_2d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            laplacianProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, result_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
                laplacianProcessBlockSlicePrePred<T>(x, size, buffer_set, cmpkit_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPostPred(
    DSize_2d& size,
    CmpBufferSet *cmpkit_set, 
    T *row_diffs, T *rowpair_diffs,
    AppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *result_pos,
    double errorBound
){
    double twice_eb = errorBound * 2;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    laplacian_compute_block_mean_difference(size.block_dim1, size.block_dim2, twice_eb, cmpkit_set->mean_quant_inds, row_diffs, rowpair_diffs);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            laplacianProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    T * row_diffs = (T *)malloc(size.block_dim1*(size.block_dim2+1)*sizeof(T));
    T * rowpair_diffs = (T *)malloc(size.block_dim2*(size.block_dim1+1)*sizeof(T));
    int * Buffer_2d = (int *)malloc(buffer_dim1 * buffer_dim2 * 3 * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    AppBufferSet_2d * buffer_set = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, blocks_mean_quant);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            laplacianProcessBlocksPostPred(size, cmpkit_set, row_diffs, rowpair_diffs, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_laplacian_2d(dim1, dim2, decData, laplacian_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(row_diffs);
    free(rowpair_diffs);
    free(Buffer_2d);
    free(decData);
}

// divergence
template <class T>
inline void divergence2DProcessBlockSlicePrePred(
    size_t x, DSize_2d size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int * vx_row_prev = vx_curr_row - off_0;
        const int * vx_row_next = vx_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            int dfxx = vx_row_next[j] - vx_row_prev[j];
            int dfyy = vy_curr_row[j+1] - vy_curr_row[j-1];
            divergence_pos[index++] = (dfxx + dfyy) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergence2DProcessBlockSlicePostPred(
    size_t x, DSize_2d size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<T *, 2>& row_diffs,
    std::array<T *, 2>& rowpair_diffs,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    T * divergence_start_pos = result_start_pos;
    const int * vx_buffer_start_pos = buffer_set[0]->currSlice_data_pos;
    const T * vx_dy_blockmean_diff = row_diffs[0] + x * (size.block_dim2 + 1);
    const T * vx_dx_top_blockmean_diff = rowpair_diffs[0] + x * size.block_dim2;
    const T * vx_dx_bott_blockmean_diff = rowpair_diffs[0] + (x + 1) * size.block_dim2;
    const int * vy_buffer_start_pos = buffer_set[1]->currSlice_data_pos;
    const T * vy_dy_blockmean_diff = row_diffs[1] + x * (size.block_dim2 + 1);
    const T * vy_dx_top_blockmean_diff = rowpair_diffs[1] + x * size.block_dim2;
    const T * vy_dx_bott_blockmean_diff = rowpair_diffs[1] + (x + 1) * size.block_dim2;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int dfxx, dfyy;
        const int * vx_curr_row = vx_buffer_start_pos;
        const int * vy_curr_row = vy_buffer_start_pos;
        T * divergence_pos = divergence_start_pos;
        for(int i=0; i<size_x; i++){
            const int * vx_prev_row = vx_curr_row - off_0;
            const int * vx_next_row = vx_curr_row + off_0;
            for(int j=0; j<size_y; j++){
                dfxx = vx_next_row[j] - vx_prev_row[j];
                dfyy = vy_curr_row[j+1] - vy_curr_row[j-1];
                divergence_pos[j] = (dfxx + dfyy) * errorBound;
            }
            divergence_pos[0] += vy_dy_blockmean_diff[y];
            divergence_pos[size_y - 1] += vy_dy_blockmean_diff[y+1];
            if(i == 0){
                for(int j=0; j<size_y; j++) divergence_pos[j] += vx_dx_top_blockmean_diff[y];
            }
            if(i == size_x-1){
                for(int j=0; j<size_y; j++) divergence_pos[j] += vx_dx_bott_blockmean_diff[y];
            }
            vx_curr_row += off_0;
            vy_curr_row += off_0;
            divergence_pos += size.offset_0;
        }
        vx_buffer_start_pos += size.Bsize;
        vy_buffer_start_pos += size.Bsize;
        divergence_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergence2DProcessBlocksPrePred(
    DSize_2d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempBlockSlice = nullptr;
    for(i=0; i<2; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks,
                            cmpkit_set[i]->mean_quant_inds, size.num_blocks);
    }
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            divergence2DProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                divergence2DProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                divergence2DProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void divergence2DProcessBlocksPostPred(
    DSize_2d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    std::array<T *, 2>& row_diffs,
    std::array<T *, 2>& rowpair_diffs,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempBlockSlice = nullptr;
    for(i=0; i<2; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks,
                            cmpkit_set[i]->mean_quant_inds, size.num_blocks);
        dxdy_compute_block_mean_difference(size.block_dim1, size.block_dim2, errorBound, cmpkit_set[i]->mean_quant_inds, row_diffs[i], rowpair_diffs[i]);
    }
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            divergence2DProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                divergence2DProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                divergence2DProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_divergence(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<int *, 2> blocks_mean_quant = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet_2d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<T *, 2> row_diffs = {nullptr, nullptr};
    std::array<T *, 2> rowpair_diffs = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        blocks_mean_quant[i] = (int *)malloc(size.num_blocks * sizeof(int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], blocks_mean_quant[i]);
        row_diffs[i] = (T *)malloc(size.block_dim1*(size.block_dim2+1)*sizeof(T));
        rowpair_diffs[i] = (T *)malloc((size.block_dim1+1)*size.block_dim2*sizeof(T));
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            divergence2DProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, row_diffs, rowpair_diffs, divergence_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            divergence2DProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_divergence_2d(dim1, dim2, decData[0], decData[1], divergence_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<2; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_2d[i]);
        free(absPredError[i]);
        free(blocks_mean_quant[i]);
        free(signFlag[i]);
        free(row_diffs[i]);
        free(rowpair_diffs[i]);
        free(decData[i]);
    }
}

// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize_2d size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curl_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int * vy_row_prev = vy_curr_row - off_0;
        const int * vy_row_next = vy_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            curl_pos[index++] = ((vy_row_next[j] - vy_row_prev[j]) - (vx_curr_row[j+1] - vx_curr_row[j-1])) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlockSlicePostPred(
    size_t x, DSize_2d size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<T *, 2>& row_diffs,
    std::array<T *, 2>& rowpair_diffs,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    T * curl_start_pos = result_start_pos;
    const int * vx_buffer_start_pos = buffer_set[0]->currSlice_data_pos;
    const T * vx_dy_blockmean_diff = row_diffs[0] + x * (size.block_dim2 + 1);
    const T * vx_dx_top_blockmean_diff = rowpair_diffs[0] + x * size.block_dim2;
    const T * vx_dx_bott_blockmean_diff = rowpair_diffs[0] + (x + 1) * size.block_dim2;
    const int * vy_buffer_start_pos = buffer_set[1]->currSlice_data_pos;
    const T * vy_dy_blockmean_diff = row_diffs[1] + x * (size.block_dim2 + 1);
    const T * vy_dx_top_blockmean_diff = rowpair_diffs[1] + x * size.block_dim2;
    const T * vy_dx_bott_blockmean_diff = rowpair_diffs[1] + (x + 1) * size.block_dim2;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int dfyx, dfxy;
        const int * vx_curr_row = vx_buffer_start_pos;
        const int * vy_curr_row = vy_buffer_start_pos;
        T * curl_pos = curl_start_pos;
        for(int i=0; i<size_x; i++){
            const int * vy_prev_row = vy_curr_row - off_0;
            const int * vy_next_row = vy_curr_row + off_0;
            for(int j=0; j<size_y; j++){
                dfyx = vy_next_row[j] - vy_prev_row[j];
                dfxy = vx_curr_row[j+1] - vx_curr_row[j-1];
                curl_pos[j] = (dfyx - dfxy) * errorBound;
            }
            curl_pos[0] -= vx_dy_blockmean_diff[y];
            curl_pos[size_y - 1] -= vx_dy_blockmean_diff[y+1];
            if(i == 0){
                for(int j=0; j<size_y; j++) curl_pos[j] += vy_dx_top_blockmean_diff[y];
            }
            if(i == size_x-1){
                for(int j=0; j<size_y; j++) curl_pos[j] += vy_dx_bott_blockmean_diff[y];
            }
            vx_curr_row += off_0;
            vy_curr_row += off_0;
            curl_pos += size.offset_0;
        }
        vx_buffer_start_pos += size.Bsize;
        vy_buffer_start_pos += size.Bsize;
        curl_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curl2DProcessBlocksPrePred(
    DSize_2d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempBlockSlice = nullptr;
    for(i=0; i<2; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks,
                            cmpkit_set[i]->mean_quant_inds, size.num_blocks);
    }
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            curlProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                curlProcessBlockSlicePrePred<T>(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void curl2DProcessBlocksPostPred(
    DSize_2d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    std::array<T *, 2>& row_diffs,
    std::array<T *, 2>& rowpair_diffs,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempBlockSlice = nullptr;
    for(i=0; i<2; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks,
                            cmpkit_set[i]->mean_quant_inds, size.num_blocks);
        dxdy_compute_block_mean_difference(size.block_dim1, size.block_dim2, errorBound, cmpkit_set[i]->mean_quant_inds, row_diffs[i], rowpair_diffs[i]);
    }
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            curlProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                curlProcessBlockSlicePostPred<T>(x, size, off_0, buffer_set, row_diffs, rowpair_diffs, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_curl(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *curl_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<int *, 2> blocks_mean_quant = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet_2d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<T *, 2> row_diffs = {nullptr, nullptr};
    std::array<T *, 2> rowpair_diffs = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        blocks_mean_quant[i] = (int *)malloc(size.num_blocks * sizeof(int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], blocks_mean_quant[i]);
        row_diffs[i] = (T *)malloc(size.block_dim1*(size.block_dim2+1)*sizeof(T));
        rowpair_diffs[i] = (T *)malloc((size.block_dim1+1)*size.block_dim2*sizeof(T));
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    }
    T * curl_pos = curl_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            curl2DProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, row_diffs, rowpair_diffs, curl_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            curl2DProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, curl_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_curl_2d(dim1, dim2, decData[0], decData[1], curl_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<2; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_2d[i]);
        free(absPredError[i]);
        free(blocks_mean_quant[i]);
        free(signFlag[i]);
        free(row_diffs[i]);
        free(rowpair_diffs[i]);
        free(decData[i]);
    }
}

#endif