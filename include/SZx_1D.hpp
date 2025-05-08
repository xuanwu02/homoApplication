#ifndef _SZX_MEAN_PREDICTOR_1D_S_HPP
#define _SZX_MEAN_PREDICTOR_1D_S_HPP

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
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * block_quant_inds = (int *)malloc(blockSideLength * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    const T * op = oriData;
    unsigned int abs_err, max_err;
    int mean_quant, curr, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            mean_quant = compute_block_mean_quant(block_size, op, block_quant_inds, inver_eb);
            op += blockSideLength;
            memcpy(outlier_pos, &mean_quant, sizeof(int));
            outlier_pos += INT_BYTES;
            max_err = 0;
            for(j=0; j<block_size; j++){
                err = block_quant_inds[j] - mean_quant;
                signFlag[j] = (err < 0);
                abs_err = abs(err);
                max_err = max_err > abs_err ? max_err : abs_err;
                absPredError[j] = abs_err;
            }
            fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            mean_quant = compute_block_mean_quant(block_size, op, block_quant_inds, inver_eb);
            memcpy(outlier_pos, &mean_quant, sizeof(int));
            outlier_pos += INT_BYTES;
            max_err = 0;
            for(j=0; j<block_size; j++){
                err = block_quant_inds[j] - mean_quant;
                signFlag[j] = (err < 0);
                abs_err = abs(err);
                max_err = max_err > abs_err ? max_err : abs_err;
                absPredError[j] = abs_err;
            }
            fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
        }
    }
    cmpSize = encode_pos - cmpData;
    free(block_quant_inds);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZx_decompress(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    double twice_eb = 2 * errorBound;
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    T * dp = decData;
    unsigned int abs_err, max_err;
    int mean_quant, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = mean_quant * twice_eb;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] = (mean_quant + err) * twice_eb;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = mean_quant * twice_eb;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] = (mean_quant + err) * twice_eb;
                }
            }
        }
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(absPredError);
    free(signFlag);
}

void SZx_decompress_meta(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * blocks_mean_quant = (int *)malloc(num_blocks * sizeof(int));
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant, err;
    size_t i, j;
    size_t block_ind = 0;
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind++] = mean_quant;
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind++] = mean_quant;
    }
    free(blocks_mean_quant);
    }
}

void SZx_decompress_postPred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * blocks_mean_quant = (int *)malloc(num_blocks * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int * dp = decData;
    unsigned int abs_err, max_err;
    int mean_quant, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind] = mean_quant;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = 0;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] =  err;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind] = mean_quant;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = 0;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] = err;
                }
            }
        }
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(blocks_mean_quant);
    free(absPredError);
    free(signFlag);
}

void SZx_decompress_prePred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * blocks_mean_quant = (int *)malloc(num_blocks * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int * dp = decData;
    unsigned int abs_err, max_err;
    int mean_quant, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind] = mean_quant;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = mean_quant;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] = mean_quant + err;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[block_ind] = mean_quant;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                for(j=0; j<block_size; j++){
                    dp[i + j] = mean_quant;
                }
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    dp[i + j] = mean_quant + err;
                }
            }
        }
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(blocks_mean_quant);
    free(absPredError);
    free(signFlag);
}

double SZx_mean_meta(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant;
    int64_t sum = 0;
    for(size_t i=0; i<num_full_block; i++){
        memcpy(&mean_quant, outlier_pos, sizeof(int));
        outlier_pos += INT_BYTES;
        sum += mean_quant * block_size;
    }
    if(num_remiander > 0){
        memcpy(&mean_quant, outlier_pos, sizeof(int));
        sum += mean_quant * num_remiander;
    }
    double mean = 2 * errorBound * sum / (double)nbEle;
    return mean;
}

double SZx_mean_postPred(
    unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
    int64_t sum = 0;
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            sum += mean_quant * block_size;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    sum += err;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            sum += mean_quant * block_size;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    sum += err;
                }
            }
        }
    }
    free(absPredError);
    free(signFlag);
    double mean = 2 * errorBound * sum / (double)nbEle;
    return mean;
}

double SZx_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant, curr, err;
    int fixed_rate;
    size_t i, j;
    size_t block_ind = 0;
    int64_t sum = 0;
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                sum += mean_quant * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    curr = mean_quant + err;
                    sum += curr;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                sum += mean_quant * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    curr = mean_quant + err;
                    sum += curr;
                }
            }
        }
    }
    free(absPredError);
    free(signFlag);
    double mean = 2 * errorBound * sum / (double)nbEle;
    return mean;
}

template <class T>
double SZx_mean_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,  size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZx_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean = -9999;
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::meta:{
            mean = SZx_mean_meta(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::postPred:{
            mean = SZx_mean_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            mean = SZx_mean_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZx_mean_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_stddev_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * blocks_mean_quant = (int *)malloc(num_blocks * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant, mean_err, err;
    int fixed_rate;
    size_t i, j;
    int64_t d;
    uint64_t d2;
    size_t block_ind = 0;
    int64_t global_mean = 0;
    int64_t global_sum = 0;
    uint64_t squared_sum = 0;
    if(num_blocks > 0){
        for(i=0; i<num_full_block; i++){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[i] = mean_quant;
            global_sum += mean_quant * block_size;
        }
    }
    if(num_remiander > 0){
        memcpy(&mean_quant, outlier_pos, sizeof(int));
        blocks_mean_quant[i] = mean_quant;
        global_sum += mean_quant * num_remiander;
    }
    global_mean = std::round((double)global_sum / (double)nbEle);
    // global_sum /= (int64_t)nbEle;
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            mean_err = blocks_mean_quant[block_ind] - global_mean;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                d = static_cast<int64_t>(mean_err);
                d2 = d * d;
                squared_sum += d2 * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    d = static_cast<int64_t>(err + mean_err);
                    d2 = d * d;
                    squared_sum += d2;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            mean_err = blocks_mean_quant[block_ind] - global_mean;
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                d = static_cast<int64_t>(mean_err);
                d2 = d * d;
                squared_sum += d2 * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    d = static_cast<int64_t>(err + mean_err);
                    d2 = d * d;
                    squared_sum += d2;
                }
            }
        }
    }
    free(blocks_mean_quant);
    free(absPredError);
    free(signFlag);
    double std = (2 * errorBound) * sqrt((double)squared_sum / (nbEle - 1));
    return std;
}

double SZx_stddev_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    size_t num_full_block = nbEle / blockSideLength;
    size_t num_remiander = nbEle % blockSideLength;
    size_t num_blocks = num_full_block + (num_remiander ? 1 : 0);
    int block_size = blockSideLength;
    int * blocks_mean_quant = (int *)malloc(num_blocks * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(blockSideLength*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(blockSideLength*sizeof(unsigned char));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * num_blocks;
    int mean_quant, curr, err;
    int fixed_rate;
    size_t i, j;
    int64_t d;
    uint64_t d2;
    size_t block_ind = 0;
    int64_t sum;
    uint64_t squared_sum = 0;
    if(num_blocks > 0){
        for(i=0; i<num_full_block; i++){
            memcpy(&mean_quant, outlier_pos, sizeof(int));
            outlier_pos += INT_BYTES;
            blocks_mean_quant[i] = mean_quant;
        }
    }
    if(num_remiander > 0){
        memcpy(&mean_quant, outlier_pos, sizeof(int));
        blocks_mean_quant[i] = mean_quant;
    }
    if(num_full_block > 0){
        for(i=0; i<nbEle-num_remiander; i+=blockSideLength){
            mean_quant = blocks_mean_quant[block_ind];
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                d = static_cast<int64_t>(mean_quant);
                d2 = d * d;
                sum += d * block_size;
                squared_sum += d2 * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    d = static_cast<int64_t>(err + mean_quant);
                    d2 = d * d;
                    sum += d;
                    squared_sum += d2;
                }
            }
        }
    }
    if(num_remiander > 0){
        block_size = num_remiander;
        for(i=nbEle-num_remiander; i<nbEle; i+=blockSideLength){
            mean_quant = blocks_mean_quant[block_ind];
            fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate == 0){
                d = static_cast<int64_t>(mean_quant);
                d2 = d * d;
                sum += d * block_size;
                squared_sum += d2 * block_size;
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                for(j=0; j<block_size; j++){
                    // if(signFlag[j] == 1) err = 0 - absPredError[j];
                    // else err = absPredError[j];
                    int s = -(int)signFlag[j];
                    err = (absPredError[j] ^ s) - s;
                    d = static_cast<int64_t>(err + mean_quant);
                    d2 = d * d;
                    sum += d;
                    squared_sum += d2;
                }
            }
        }
    }
    free(blocks_mean_quant);
    free(absPredError);
    free(signFlag);
    double std = (2 * errorBound) * sqrt(((double)squared_sum - (double)sum * sum / nbEle) / (nbEle - 1));
    return std;
}

template <class T>
double SZx_stddev_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double std = -9999;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            std = SZx_stddev_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            std = SZx_stddev_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            std = SZx_stddev_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return std;
}

template <class T>
void recoverBlockSlice2PrePred(
    size_t x, T& size, size_t& offset,
    size_t& block_ind, size_t& accumulated_num, CmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, unsigned char *& outlier_pos,
    int *currSlice_data_pos, int *prevSlice_data_pos,
    size_t offset_0
){
clock_gettime(CLOCK_REALTIME, &start2);
    if(offset) memcpy(currSlice_data_pos, prevSlice_data_pos+size.Bwidth*offset_0, offset*sizeof(int));
    int mean_quant, err;
    int fixed_rate;
    size_t j, index = 0;
    size_t target_num = (x + 1) * size.Bwidth * size.offset_0;
    if(target_num > size.nbEle) target_num = size.nbEle;
    int * data_pos = currSlice_data_pos + offset;
    while(accumulated_num < target_num){
        int block_size = ((block_ind + 1) * size.Bsize < size.nbEle) ? size.Bsize : (size.nbEle - block_ind * size.Bsize);
        memcpy(&mean_quant, outlier_pos, sizeof(int));
        outlier_pos += INT_BYTES;
        fixed_rate = (int)cmpkit_set->compressed[block_ind++];
        if(fixed_rate == 0){
            for(j=0; j<block_size; j++){
                data_pos[index++] = mean_quant;
            }
        }else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            for(j=0; j<block_size; j++){
                // if(cmpkit_set->signFlag[j] == 1) err = 0 - cmpkit_set->absPredError[j];
                // else err = cmpkit_set->absPredError[j];
                int s = -(int)cmpkit_set->signFlag[j];
                err = (cmpkit_set->absPredError[j] ^ s) - s;
                data_pos[index++] = mean_quant + err;
            }
        }
        accumulated_num += size.Bsize;
    }
    index += offset;
    offset = index % size.offset_0;
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockSlicePrePred(
    size_t x, DSize2D_1d& size, AppBufferSet2D_1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    int index = 0;
    const int * curr_row = buffer_set->currSlice_data_pos;
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
inline void dxdyProcessBlocksPrePred(
    DSize2D_1d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet2D_1d *buffer_set,
    unsigned char *&encode_pos,
    unsigned char *&outlier_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t rec_offset = 0, block_ind = 0;
    size_t accumulated_num = 0;
    size_t BlockSliceSize = size.Bwidth * size.dim2;
    buffer_set->reset();
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempSlice_pos);
            if(x == size.num_blockSlice - 1){
                dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
                dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_dxdy(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    assert(dim3 == 1);
    DSize2D_1d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 3 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.Bsize*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.Bsize*sizeof(unsigned char));
    AppBufferSet2D_1d * buffer_set = new AppBufferSet2D_1d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(Buffer_2d);
    free(absPredError);
    free(signFlag);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize2D_1d& size,
    AppBufferSet2D_1d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * result_pos = result_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            result_pos[index++] = (curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j]) * errorBound * 2;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize2D_1d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet2D_1d *buffer_set,
    unsigned char *&encode_pos,
    unsigned char *&outlier_pos,
    T *result_pos,
    double errorBound
){
    size_t rec_offset = 0, block_ind = 0;
    size_t accumulated_num = 0;
    size_t BlockSliceSize = size.Bwidth * size.dim2;
    buffer_set->reset();
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempSlice_pos);
            if(x == size.num_blockSlice - 1){
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian2D(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    assert(dim3 == 1);
    DSize2D_1d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 3 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.Bsize*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.Bsize*sizeof(unsigned char));
    AppBufferSet2D_1d * buffer_set = new AppBufferSet2D_1d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(Buffer_2d);
    free(absPredError);
    free(signFlag);
    free(decData);
}

// divergence
template <class T>
inline void divergenceProcessBlockSlicePrePred(
    size_t x, DSize2D_1d& size,
    std::array<AppBufferSet2D_1d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx = buffer_set[0]->currSlice_data_pos;
    const int * vy = buffer_set[1]->currSlice_data_pos;
    auto idx = [&](size_t i, size_t j) {
        return i * size.dim2 + j;
    };
    for (size_t i=0; i<size_x; i++) {
        for (size_t j=0; j<size.dim2; j++) {
            int dfxx = vx[idx(i + 1, j)] - vx[idx(i - 1, j)];
            int dfyy = vy[idx(i, j + 1)] - vy[idx(i, j - 1)];
            divergence_pos[idx(i, j)] = (dfxx + dfyy) * errorBound;
        }
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlocksPrePred(
    DSize2D_1d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet2D_1d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    std::array<unsigned char *, 2>& outlier_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bwidth * size.dim2;
    std::array<size_t, 2> rec_offset = {0, 0};
    std::array<size_t, 2> block_ind = {0, 0};
    std::array<size_t, 2> accumulated_num = {0, 0};
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
            }
            divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.num_blockSlice - 1){
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
                }
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_divergence2D(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    assert(dim3 == 1);
    DSize2D_1d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet2D_1d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    std::array<unsigned char *, 2> outlier_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 3 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.Bsize * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.Bsize * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet2D_1d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
        outlier_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            divergenceProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
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
        free(signFlag[i]);
        free(decData[i]);
    }
}

// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize2D_1d& size,
    std::array<AppBufferSet2D_1d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curl_pos = result_start_pos;
    const int * vx = buffer_set[0]->currSlice_data_pos;
    const int * vy = buffer_set[1]->currSlice_data_pos;
    auto idx = [&](size_t i, size_t j) {
        return i * size.dim2 + j;
    };
    for (size_t i=0; i<size_x; i++) {
        for (size_t j=0; j<size.dim2; j++) {
            curl_pos[idx(i, j)] = ((vy[idx(i + 1, j)] - vy[idx(i - 1, j)]) - (vx[idx(i, j + 1)] - vx[idx(i, j - 1)])) * errorBound;
        }
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlocksPrePred(
    DSize2D_1d& size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet2D_1d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    std::array<unsigned char *, 2>& outlier_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bwidth * size.dim2;
    std::array<size_t, 2> rec_offset = {0, 0};
    std::array<size_t, 2> block_ind = {0, 0};
    std::array<size_t, 2> accumulated_num = {0, 0};
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
            }
            curlProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.num_blockSlice - 1){
                curlProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
                }
                curlProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_curl2D(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *curl_result, decmpState state
){
    assert(dim3 == 1);
    DSize2D_1d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet2D_1d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    std::array<unsigned char *, 2> outlier_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 3 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.Bsize * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.Bsize * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet2D_1d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
        outlier_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * curl_pos = curl_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            curlProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, curl_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
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
        free(signFlag[i]);
        free(decData[i]);
    }
}

template <class T>
inline void dxdydzProcessBlockSlicePrePred(
    size_t x, DSize3D_1d& size, AppBufferSet3D_1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * prevBlockSliceBottom_pos = buffer_set->prevSlice_data_pos + (size.Bwidth - 1) * buffer_set->offset_0;
    const int * nextBlockSliceTop_pos = buffer_set->nextSlice_data_pos;
    if(!isTopSlice) memcpy(buffer_set->currSlice_data_pos-buffer_set->offset_0, prevBlockSliceBottom_pos, buffer_set->offset_0*sizeof(int));
    if(!isBottomSlice) memcpy(buffer_set->currSlice_data_pos+size.Bwidth*buffer_set->offset_0, nextBlockSliceTop_pos, buffer_set->offset_0*sizeof(int));
    const int * curr_plane = buffer_set->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.offset_0;
        T * x_dy_pos = dy_start_pos + i * size.offset_0;
        T * x_dz_pos = dz_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.offset_1;
            T * y_dy_pos = x_dy_pos + j * size.offset_1;
            T * y_dz_pos = x_dz_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index_2d = j * buffer_set->offset_1 + k;
                y_dx_pos[k] = (next_plane[buffer_index_2d] - prev_plane[buffer_index_2d]) * errorBound;
                y_dy_pos[k] = (next_row[k] - prev_row[k]) * errorBound;
                y_dz_pos[k] = (curr_row[k + 1] - curr_row[k - 1]) * errorBound;
            }
            curr_row += buffer_set->offset_1;
        }
        curr_plane += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize3D_1d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet3D_1d *buffer_set,
    unsigned char *&encode_pos,
    unsigned char *&outlier_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t rec_offset = 0, block_ind = 0;
    size_t accumulated_num = 0;
    size_t BlockSliceSize = size.Bwidth * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.num_blockSlice - 1){
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_dxdydz(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound, T *dx_result,
    T *dy_result, T *dz_result, decmpState state
){
    DSize3D_1d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_dim3 = size.dim3;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.Bsize*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.Bsize*sizeof(unsigned char));
    AppBufferSet3D_1d * buffer_set = new AppBufferSet3D_1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_dxdydz(dim1, dim2, dim3, decData, dx_pos, dy_pos, dz_pos);
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
    free(Buffer_3d);
    free(absPredError);
    free(signFlag);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize3D_1d& size, AppBufferSet3D_1d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * curr_plane = buffer_set->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        T * laplacian_pos = result_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            T * y_laplacian_pos = laplacian_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                size_t buffer_index_2d = j * buffer_set->offset_1 + k;
                y_laplacian_pos[index_1d] = (curr_row[k-1] + curr_row[k+1] +
                                             prev_row[k] + next_row[k] +
                                             prev_plane[buffer_index_2d] + next_plane[buffer_index_2d] -
                                             6 * curr_row[k]) * errorBound * 2;
            }
            curr_row += buffer_set->offset_1;
        }
        curr_plane += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize3D_1d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet3D_1d *buffer_set,
    unsigned char *&encode_pos,
    unsigned char *&outlier_pos,
    T *result_start_pos,
    double errorBound
){
    size_t rec_offset = 0, block_ind = 0;
    size_t accumulated_num = 0;
    size_t BlockSliceSize = size.Bwidth * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_start_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.num_blockSlice - 1){
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_start_pos+offset, errorBound, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, rec_offset, block_ind, accumulated_num, cmpkit_set, encode_pos, outlier_pos, buffer_set->nextSlice_data_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_start_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian3D(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize3D_1d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_dim3 = size.dim3;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.Bsize*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.Bsize*sizeof(unsigned char));
    AppBufferSet3D_1d * buffer_set = new AppBufferSet3D_1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned char * outlier_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_laplacian_3d(dim1, dim2, dim3, decData, laplacian_pos);
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
    free(Buffer_3d);
    free(absPredError);
    free(signFlag);
    free(decData);
}

// divergence
template <class T>
inline void divergenceProcessBlockSlicePrePred(
    size_t x, DSize3D_1d& size,
    std::array<AppBufferSet3D_1d *, 3>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx = buffer_set[0]->currSlice_data_pos;
    const int * vy = buffer_set[1]->currSlice_data_pos;
    const int * vz = buffer_set[2]->currSlice_data_pos;
    auto idx = [&](size_t i, size_t j, size_t k) {
        return i * size.dim2 * size.dim3 + j * size.dim3 + k;
    };
    for (size_t i=0; i<size_x; i++) {
        for (size_t j=0; j<size.dim2; j++) {
            for (size_t k=0; k<size.dim3; k++) {
                int dfxx = vx[idx(i + 1, j, k)] - vx[idx(i - 1, j, k)];
                int dfyy = vy[idx(i, j + 1, k)] - vy[idx(i, j - 1, k)];
                int dfzz = vz[idx(i, j, k + 1)] - vz[idx(i, j, k - 1)];
                divergence_pos[idx(i, j, k)] = (dfxx + dfyy + dfzz) * errorBound;
            }
        }
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlocksPrePred(
    DSize3D_1d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet3D_1d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    std::array<unsigned char *, 3>& outlier_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bwidth * size.dim2 * size.dim3;
    std::array<size_t, 3> rec_offset = {0, 0, 0};
    std::array<size_t, 3> block_ind = {0, 0, 0};
    std::array<size_t, 3> accumulated_num = {0, 0, 0};
    for(i=0; i<3; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
            }
            divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.num_blockSlice - 1){
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
                }
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_divergence3D(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    DSize3D_1d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_dim3 = size.dim3;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet3D_1d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> outlier_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc(buffer_size * 3 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.Bsize * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.Bsize * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet3D_1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
        outlier_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            divergenceProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<3; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_divergence_3d(dim1, dim2, dim3, decData[0], decData[1], decData[2], divergence_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<3; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_3d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}


// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize3D_1d& size,
    std::array<AppBufferSet3D_1d *, 3>& buffer_set,
    T *curlx_start_pos, T *curly_start_pos, T *curlz_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    buffer_set[2]->setGhostEle(size, isTopSlice, isBottomSlice);
    T * curlx_pos = curlx_start_pos;
    T * curly_pos = curly_start_pos;
    T * curlz_pos = curlz_start_pos;
    const int * vx = buffer_set[0]->currSlice_data_pos;
    const int * vy = buffer_set[1]->currSlice_data_pos;
    const int * vz = buffer_set[2]->currSlice_data_pos;
    auto idx = [&](size_t i, size_t j, size_t k) {
        return i * size.dim2 * size.dim3 + j * size.dim3 + k;
    };
    for (size_t i=0; i<size_x; i++) {
        for (size_t j=0; j<size.dim2; j++) {
            for (size_t k=0; k<size.dim3; k++) {
                curlx_pos[idx(i, j, k)] = ((vz[idx(i, j + 1, k)] - vz[idx(i, j - 1, k)]) - (vy[idx(i, j, k + 1)] - vy[idx(i, j, k - 1)])) * errorBound;
                curly_pos[idx(i, j, k)] = ((vx[idx(i, j, k + 1)] - vx[idx(i, j, k - 1)]) - (vz[idx(i + 1, j, k)] - vz[idx(i - 1, j, k)])) * errorBound;
                curlz_pos[idx(i, j, k)] = ((vy[idx(i + 1, j, k)] - vy[idx(i - 1, j, k)]) - (vx[idx(i, j + 1, k)] - vx[idx(i, j - 1, k)])) * errorBound;
            }
        }
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlocksPrePred(
    DSize3D_1d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet3D_1d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    std::array<unsigned char *, 3>& outlier_pos,
    T *curlx_pos, T *curly_pos, T *curlz_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bwidth * size.dim2 * size.dim3;
    std::array<size_t, 3> rec_offset = {0, 0, 0};
    std::array<size_t, 3> block_ind = {0, 0, 0};
    std::array<size_t, 3> accumulated_num = {0, 0, 0};
    for(i=0; i<3; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.num_blockSlice; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, off_0);
                recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
            }
            curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.num_blockSlice - 1){
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PrePred(x+1, size, rec_offset[i], block_ind[i], accumulated_num[i], cmpkit_set[i], encode_pos[i], outlier_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i]->currSlice_data_pos, off_0);
                }
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_curl3D(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *curlx_result, T *curly_result, T *curlz_result,
    decmpState state
){
    DSize3D_1d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2;
    size_t buffer_dim3 = size.dim3;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet3D_1d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> outlier_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc(buffer_size * 3 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.Bsize * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.Bsize * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet3D_1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
        outlier_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * curlx_pos = curlx_result;
    T * curly_pos = curly_result;
    T * curlz_pos = curlz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            curlProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, outlier_pos, curlx_pos, curly_pos, curlz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<3; i++){
                SZx_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_curl_3d(dim1, dim2, dim3, decData[0], decData[1], decData[2], curlx_pos, curly_pos, curlz_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<3; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_3d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}

#endif