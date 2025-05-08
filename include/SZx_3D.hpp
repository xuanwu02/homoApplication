#ifndef _SZX_MEAN_PREDICTOR_3D_HPP
#define _SZX_MEAN_PREDICTOR_3D_HPP

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
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * block_quant_inds = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            const T * z_data_pos = y_data_pos;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                unsigned int * abs_err_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                int fixed_rate, max_err = 0;
                int mean_quant = compute_block_mean_quant(size_x, size_y, size_z, size.offset_0, size.offset_1, z_data_pos, block_quant_inds, inver_eb);
                int * block_buffer_pos = block_quant_inds;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            int err = *block_buffer_pos++ - mean_quant;
                            int abs_err = abs(err);
                            *sign_pos++ = (err < 0);
                            *abs_err_pos++ = abs_err;
                            max_err = max_err > abs_err ? max_err : abs_err;
                        }
                    }
                }
                fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                memcpy(qmean_pos, &mean_quant, sizeof(int));
                qmean_pos += 4;
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                    encode_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                    encode_pos += savedbitsbyteLength;
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.offset_1;
        }
        x_data_pos += size.Bsize * size.offset_0;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(block_quant_inds);
}

template <class T>
void SZx_decompress(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    double twice_eb = 2 * errorBound;
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            T * z_data_pos = y_data_pos;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int mean_quant = blocks_mean_quant[block_ind];
                int fixed_rate = (int)cmpData[block_ind++];
                T * curr_data_pos = z_data_pos;
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
                            for(int k=0; k<size_z; k++){
                                int s = -(int)signFlag[index];
                                curr = (absPredError[index] ^ s) - s;
                                index++;
                                *curr_data_pos++ = (curr + mean_quant) * twice_eb;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = mean_quant * twice_eb;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.offset_1;
        }
        x_data_pos += size.Bsize * size.offset_0;
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_meta(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    free(blocks_mean_quant);
}

void SZx_decompress_postPred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int * z_data_pos = y_data_pos;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                int * curr_data_pos = z_data_pos;
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
                            for(int k=0; k<size_z; k++){
                                int s = -(int)signFlag[index];
                                curr = (absPredError[index] ^ s) - s;
                                index++;
                                *curr_data_pos++ = curr;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = 0;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.offset_1;
        }
        x_data_pos += size.Bsize * size.offset_0;
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_prePred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int * z_data_pos = y_data_pos;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int mean_quant = blocks_mean_quant[block_ind];
                int fixed_rate = (int)cmpData[block_ind++];
                int * curr_data_pos = z_data_pos;
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
                            for(int k=0; k<size_z; k++){
                                int s = -(int)signFlag[index];
                                curr = (absPredError[index] ^ s) - s;
                                index++;
                                *curr_data_pos++ = curr + mean_quant;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = mean_quant;
                            }
                            curr_data_pos += size.offset_1 - size_z;
                        }
                        curr_data_pos += size.offset_0 - size_y * size.offset_1;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.offset_1;
        }
        x_data_pos += size.Bsize * size.offset_0;
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

double SZx_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int64_t quant_sum = 0;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
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
                        int s = -(int)signFlag[i];
                        curr = (absPredError[i] ^ s) - s;
                        curr += mean_quant;
                        quant_sum += curr;
                    }
                }else{
                    quant_sum += mean_quant * block_size;
                }
            }
        }
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

double SZx_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int64_t quant_sum = 0;
    int block_ind = 0;
    extract_block_mean(cmpData+size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
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
                        int s = -(int)signFlag[i];
                        curr = (absPredError[i] ^ s) - s;
                        quant_sum += curr;
                    }
                }
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

double SZx_mean_meta(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int64_t sum = 0;
    int mean_quant;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                memcpy(&mean_quant, qmean_pos, sizeof(int));
                qmean_pos += INT_BYTES;
                sum += mean_quant * block_size;
            }
        }
    }
    double mean = 2 * errorBound * sum / size.nbEle;
    return mean;
}

template <class T>
double SZx_mean_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
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
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZx_mean_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            mean = SZx_mean_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_mean_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::meta:{
            mean = SZx_mean_meta(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_region_mean_meta(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t dlo3 = floor(dim3 * (1.0 - ratio) * 0.5);
    size_t dhi3 = floor(dim3 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t lo3 = dlo3 / size.Bsize;
    size_t hi3 = dhi3 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * (hi3 - lo3) * size.Bsize * size.Bsize * size.Bsize;
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    int block_ind, mean_quant;
    int size_x, size_y, size_z, block_size;
    size_t x, y, z;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_ind = x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + lo3;
            for(z=lo3; z<hi3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                mean_quant = blocks_mean_quant[block_ind++];
                quant_sum += mean_quant * block_size;
            }
        }
    }
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

double SZx_region_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t dlo3 = floor(dim3 * (1.0 - ratio) * 0.5);
    size_t dhi3 = floor(dim3 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t lo3 = dlo3 / size.Bsize;
    size_t hi3 = dhi3 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * (hi3 - lo3) * size.Bsize * size.Bsize * size.Bsize;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    const unsigned char * rate_start_pos = cmpData;
    unsigned char * encode_start_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    std::vector<std::vector<size_t>> prefix(hi1 - lo1, std::vector<size_t>(hi2 - lo2,  0));
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y, z, i, j;
    size_t byteLengthPrefix = 0;
    int block_ind = 0;
    int size_x, size_y, size_z, block_size, fixed_rate;
    int mean_quant, curr;
    for(x=0; x<lo1; x++){
        for(y=0; y<size.block_dim2; y++){
            for(z=0; z<size.block_dim3; z++){
                size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
        }
    }
    i = 0;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        j = 0;
        for(y=0; y<lo2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<size.block_dim3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }                
            }
        }
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<lo3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
            prefix[i][j++] = byteLengthPrefix;
            for(z=lo3; z<size.block_dim3; z++){
                size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
        }
        for(y=hi2; y<size.block_dim2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<size.block_dim3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }                
            }
        }
        i++;
    }
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_ind = x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + lo3;
            unsigned char * encode_pos = encode_start_pos + prefix[x - lo1][y - lo2];
            for(z=lo3; z<hi3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
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
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

double SZx_region_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    double ratio, int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t dlo3 = floor(dim3 * (1.0 - ratio) * 0.5);
    size_t dhi3 = floor(dim3 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t lo3 = dlo3 / size.Bsize;
    size_t hi3 = dhi3 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * (hi3 - lo3) * size.Bsize * size.Bsize * size.Bsize;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    const unsigned char * rate_start_pos = cmpData;
    unsigned char * encode_start_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    std::vector<std::vector<size_t>> prefix(hi1 - lo1, std::vector<size_t>(hi2 - lo2,  0));
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    int64_t quant_sum = 0;
    size_t x, y, z, i, j;
    size_t byteLengthPrefix = 0;
    int block_ind = 0;
    int size_x, size_y, size_z, block_size, fixed_rate;
    int mean_quant, curr;
    for(x=0; x<lo1; x++){
        for(y=0; y<size.block_dim2; y++){
            for(z=0; z<size.block_dim3; z++){
                size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
        }
    }
    i = 0;
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        j = 0;
        for(y=0; y<lo2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<size.block_dim3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }                
            }
        }
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<lo3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
            prefix[i][j++] = byteLengthPrefix;
            for(z=lo3; z<size.block_dim3; z++){
                size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }
            }
        }
        for(y=hi2; y<size.block_dim2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(z=0; z<size.block_dim3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
                fixed_rate = (int)rate_start_pos[block_ind++];
                if(fixed_rate){
                    byteLengthPrefix += (block_size + 7) / 8 + getByteLength(block_size, fixed_rate);
                }                
            }
        }
        i++;
    }
    for(x=lo1; x<hi1; x++){
        size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(y=lo2; y<hi2; y++){
            size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            block_ind = x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + lo3;
            unsigned char * encode_pos = encode_start_pos + prefix[x - lo1][y - lo2];
            for(z=lo3; z<hi3; z++){
                size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                block_size = size_x * size_y * size_z;
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
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / region_size;
    return mean;
}

template <class T>
double SZx_region_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    double ratio, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            SZx_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            mean = compute_region_mean(dim1, dim2, dim3, blockSideLength, ratio, decData);
            break;
        }
        case decmpState::prePred:{
            mean = SZx_region_mean_prePred(cmpData, dim1, dim2, dim3, ratio, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_region_mean_postPred(cmpData, dim1, dim2, dim3, ratio, blockSideLength, errorBound);            
            break;
        }
        case decmpState::meta:{
            mean = SZx_region_mean_meta(cmpData, dim1, dim2, dim3, ratio, blockSideLength, errorBound);            
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
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_3d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    uint64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                uint64_t block_size = size_x * size_y * size_z;
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
                        if(signFlag[index]) curr = 0 - absPredError[index];
                        else curr = absPredError[index];
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
    }
    free(absPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double std = (2 * errorBound) * sqrt((double)squared_sum / (size.nbEle - 1));
    return std;
}

double SZx_stddev_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    extract_block_mean(qmean_pos, blocks_mean_quant, size.num_blocks);
    int block_ind = 0;
    int64_t d;
    int64_t quant_sum = 0;
    uint64_t d2;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                uint64_t block_size = size_x * size_y * size_z;
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
                        if(signFlag[index]) curr = 0 - absPredError[index];
                        else curr = absPredError[index];
                        index++;
                        int curr_quant = curr + block_mean;
                        d = static_cast<int64_t>(curr + block_mean);
                        d2 = d * d;
                        quant_sum += d;
                        squared_quant_sum += d2;
                    }
                }else{
                    d = static_cast<int64_t>(block_mean);
                    d2 = d * d;
                    quant_sum += d * block_size;
                    squared_quant_sum += d2 * block_size;
                }
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
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double std;

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

inline void recoverBlockSlice2PrePred(
    size_t x, DSize_3d size, CmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t offset_0, size_t offset_1
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * curr_buffer_pos = buffer_start_pos;
            int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            int s = -(int)cmpkit_set->signFlag[index];
                            curr = (cmpkit_set->absPredError[index] ^ s) - s;
                            index++;
                            curr_buffer_pos[k] = curr + mean_quant;
                        }
                        curr_buffer_pos += offset_1;
                    }
                    curr_buffer_pos += offset_0 - size_y * offset_1;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] =  mean_quant;
                        }
                        curr_buffer_pos += offset_1;
                    }
                    curr_buffer_pos += offset_0 - size_y * offset_1;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockSlice2PostPred(
    size_t x, DSize_3d size, CmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t offset_0, size_t offset_1
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * curr_buffer_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            int s = -(int)cmpkit_set->signFlag[index];
                            curr_buffer_pos[k] = (cmpkit_set->absPredError[index] ^ s) - s;
                            index++;
                        }
                        curr_buffer_pos += offset_1;
                    }
                    curr_buffer_pos += offset_0 - size_y * offset_1;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] =  0;
                        }
                        curr_buffer_pos += offset_1;
                    }
                    curr_buffer_pos += offset_0 - size_y * offset_1;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlockSlicePrePred(
    size_t x, DSize_3d size,
    AppBufferSet_3d *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * curr_plane = buffer_set->currSlice_data_pos - buffer_set->offset_1 - 1;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.offset_0;
        T * x_dy_pos = dy_start_pos + i * size.offset_0;
        T * x_dz_pos = dz_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane + buffer_set->offset_1 + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.offset_1;
            T * y_dy_pos = x_dy_pos + j * size.offset_1;
            T * y_dz_pos = x_dz_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index_2d = (j + 1) * buffer_set->offset_1 + k + 1;
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
inline void dxdydzProcessBlockSlicePostPred(
    size_t x, DSize_3d size, AppBufferSet_3d *buffer_set,
    double errorBound, T *x_diffs, T *y_diffs, T *z_diffs,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    auto idx = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_y = [&](size_t x, size_t y, size_t z)
                 { return x * (size.block_dim2+1) * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_z = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * (size.block_dim3+1) + y * (size.block_dim3+1) + z; };
    const int * y_buffer_pos = buffer_set->currSlice_data_pos;
    T * y_dx_pos = dx_start_pos;
    T * y_dy_pos = dy_start_pos;
    T * y_dz_pos = dz_start_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        const int * z_buffer_pos = y_buffer_pos;
        T * z_dx_pos = y_dx_pos;
        T * z_dy_pos = y_dy_pos;
        T * z_dz_pos = y_dz_pos;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            T * dx_pos = z_dx_pos;
            T * dy_pos = z_dy_pos;
            T * dz_pos = z_dz_pos;
            const int * curr_row_pos = z_buffer_pos;
            const int * prev_row_pos = z_buffer_pos - buffer_set->offset_1;
            const int * next_row_pos = z_buffer_pos + buffer_set->offset_1;
            const int * top_row_pos = z_buffer_pos - buffer_set->offset_0;
            const int * bott_row_pos = z_buffer_pos + buffer_set->offset_0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        dx_pos[k] = (bott_row_pos[k] - top_row_pos[k]) * errorBound;
                        dy_pos[k] = (next_row_pos[k] - prev_row_pos[k]) * errorBound;
                        dz_pos[k] = (curr_row_pos[k+1] - curr_row_pos[k-1]) * errorBound;
                    }
                    if(i == 0){
                        for(int k=0; k<size_z; k++){
                            dx_pos[k] += x_diffs[idx(x,y,z)];
                        }
                    }
                    else if(i == size_x-1){
                        for(int k=0; k<size_z; k++){
                            dx_pos[k] += x_diffs[idx(x+1,y,z)];
                        }
                    }
                    if(j == 0){
                        for(int k=0; k<size_z; k++){
                            dy_pos[k] += y_diffs[idx_y(x,y,z)];
                        }
                    }
                    else if(j == size_y-1){
                        for(int k=0; k<size_z; k++){
                            dy_pos[k] += y_diffs[idx_y(x,y+1,z)];
                        }
                    }
                    dz_pos[0] += z_diffs[idx_z(x,y,z)];
                    dz_pos[size_z-1] += z_diffs[idx_z(x,y,z+1)];
                    curr_row_pos += buffer_set->offset_1;
                    prev_row_pos += buffer_set->offset_1;
                    next_row_pos += buffer_set->offset_1;
                    top_row_pos += buffer_set->offset_1;
                    bott_row_pos += buffer_set->offset_1;
                    dx_pos += size.offset_1;
                    dy_pos += size.offset_1;
                    dz_pos += size.offset_1;
                }
                curr_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                prev_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                next_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                top_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                bott_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                dx_pos += size.offset_0 - size_y * size.offset_1;
                dy_pos += size.offset_0 - size_y * size.offset_1;
                dz_pos += size.offset_0 - size_y * size.offset_1;
            }
            z_buffer_pos += size.Bsize;
            z_dx_pos += size.Bsize;
            z_dy_pos += size.Bsize;
            z_dz_pos += size.Bsize;
        }
        y_buffer_pos += size.Bsize * buffer_set->offset_1;
        y_dx_pos += size.Bsize * size.offset_1;
        y_dy_pos += size.Bsize * size.offset_1;
        y_dz_pos += size.Bsize * size.offset_1;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            dxdydzProcessBlockSlicePrePred(x, size, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdydzProcessBlocksPostPred(
    DSize_3d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    unsigned char *encode_pos,
    T *x_diffs, T *y_diffs, T *z_diffs,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    dxdydz_compute_block_mean_difference(size.block_dim1, size.block_dim2, size.block_dim3, errorBound, cmpkit_set->mean_quant_inds, x_diffs, y_diffs, z_diffs);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            dxdydzProcessBlockSlicePostPred(x, size, buffer_set, errorBound, x_diffs, y_diffs, z_diffs, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockSlicePostPred(x, size, buffer_set, errorBound, x_diffs, y_diffs, z_diffs, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
                dxdydzProcessBlockSlicePostPred(x, size, buffer_set, errorBound, x_diffs, y_diffs, z_diffs, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
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
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    T * x_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    T * y_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    T * z_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    AppBufferSet_3d * buffer_set = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, blocks_mean_quant);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdydzProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, x_diffs, y_diffs, z_diffs, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
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
    free(x_diffs);
    free(y_diffs);
    free(z_diffs);
    free(blocks_mean_quant);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize_3d size, CmpBufferSet *cmpkit_set,
    AppBufferSet_3d *buffer_set, double errorBound,
    T *result_start_pos, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * curr_plane = buffer_set->currSlice_data_pos - buffer_set->offset_1 - 1;
    for(int i=0; i<size_x; i++){
        T * x_result_pos = result_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane + buffer_set->offset_1 + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_result_pos = x_result_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index_2d = (j + 1) * buffer_set->offset_1 + k + 1;
                y_result_pos[k] = (curr_row[k - 1] + curr_row[k + 1] +
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
inline void laplacianProcessBlockSlicePostPred(
    size_t x, DSize_3d size, AppBufferSet_3d *buffer_set,
    double twice_eb, T *x_diffs, T *y_diffs, T *z_diffs,
    T *result_start_pos, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    auto idx = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_y = [&](size_t x, size_t y, size_t z)
                 { return x * (size.block_dim2+1) * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_z = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * (size.block_dim3+1) + y * (size.block_dim3+1) + z; };
    const int * y_buffer_pos = buffer_set->currSlice_data_pos;
    T * y_result_pos = result_start_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        const int * z_buffer_pos = y_buffer_pos;
        T * z_result_pos = y_result_pos;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            T * result_pos = z_result_pos;
            const int * curr_row_pos = z_buffer_pos;
            const int * prev_row_pos = z_buffer_pos - buffer_set->offset_1;
            const int * next_row_pos = z_buffer_pos + buffer_set->offset_1;
            const int * top_row_pos = z_buffer_pos - buffer_set->offset_0;
            const int * bott_row_pos = z_buffer_pos + buffer_set->offset_0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        result_pos[k] = (top_row_pos[k] + bott_row_pos[k] + 
                                        prev_row_pos[k] + next_row_pos[k] + 
                                        curr_row_pos[k-1] + curr_row_pos[k+1] -
                                        6 * curr_row_pos[k]) * twice_eb;
                    }
                    if(i == 0){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] -= x_diffs[idx(x,y,z)];
                        }
                    }
                    else if(i == size_x-1){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += x_diffs[idx(x+1,y,z)];
                        }
                    }
                    if(j == 0){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] -= y_diffs[idx_y(x,y,z)];
                        }
                    }
                    else if(j == size_y-1){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += y_diffs[idx_y(x,y+1,z)];
                        }
                    }
                    result_pos[0] -= z_diffs[idx_z(x,y,z)];
                    result_pos[size_z-1] += z_diffs[idx_z(x,y,z+1)];
                    curr_row_pos += buffer_set->offset_1;
                    prev_row_pos += buffer_set->offset_1;
                    next_row_pos += buffer_set->offset_1;
                    top_row_pos += buffer_set->offset_1;
                    bott_row_pos += buffer_set->offset_1;
                    result_pos += size.offset_1;
                }
                curr_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                prev_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                next_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                top_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                bott_row_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                result_pos += size.offset_0 - size_y * size.offset_1;
            }
            z_buffer_pos += size.Bsize;
            z_result_pos += size.Bsize;
        }
        y_buffer_pos += size.Bsize * buffer_set->offset_1;
        y_result_pos += size.Bsize * size.offset_1;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize_3d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    unsigned char *encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            laplacianProcessBlockSlicePrePred(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePrePred(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, false, true);
            }else{
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
                laplacianProcessBlockSlicePrePred(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPostPred(
    DSize_3d& size,
    CmpBufferSet *cmpkit_set,
    T *x_diffs, T *y_diffs, T *z_diffs,
    AppBufferSet_3d *buffer_set,
    unsigned char *encode_pos,
    T *result_pos,
    double errorBound
){
    double twice_eb = errorBound * 2;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockSlice = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    laplacian_compute_block_mean_difference(size.block_dim1, size.block_dim2, size.block_dim3, twice_eb, cmpkit_set->mean_quant_inds, x_diffs, y_diffs, z_diffs);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
            laplacianProcessBlockSlicePostPred(x, size, buffer_set, twice_eb, x_diffs, y_diffs, z_diffs, result_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, twice_eb, x_diffs, y_diffs, z_diffs, result_pos+offset, false, true);
            }else{
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0, buffer_set->offset_1);
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, twice_eb, x_diffs, y_diffs, z_diffs, result_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    T * x_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    T * y_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    T * z_diffs = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    AppBufferSet_3d * buffer_set = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, blocks_mean_quant);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            laplacianProcessBlocksPostPred(size, cmpkit_set, x_diffs, y_diffs, z_diffs, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
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
    free(x_diffs);
    free(y_diffs);
    free(z_diffs);
    free(blocks_mean_quant);
}

// divergence
template <class T>
inline void divergenceProcessBlockSlicePrePred(
    size_t x, DSize_3d size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    T *result_start_pos, double errorBound,
    size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    T * divergence_pos = result_start_pos;
    size_t index = 0;
    const int * vx_curr_plane = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_plane = buffer_set[1]->currSlice_data_pos;
    const int * vz_curr_plane = buffer_set[2]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int* vx_plane_prev = vx_curr_plane - off_0;
        const int* vx_plane_next = vx_curr_plane + off_0;
        const int* vy_plane_base = vy_curr_plane;
        const int* vz_plane_base = vz_curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            size_t row_offset = j * off_1;
            const int* vx_row_prev = vx_plane_prev + row_offset;
            const int* vx_row_next = vx_plane_next + row_offset;
            const int* vy_row_prev = vy_plane_base - off_1;
            const int* vy_row_next = vy_plane_base + off_1;
            const int* vz_row = vz_plane_base;
            for(size_t k=0; k<size.dim3; k++){
                int dfxx = vx_row_next[k] - vx_row_prev[k];
                int dfyy = vy_row_next[k] - vy_row_prev[k];
                int dfzz = vz_row[k+1] - vz_row[k-1];
                divergence_pos[index++] = (dfxx + dfyy + dfzz) * errorBound;
            }
            vy_plane_base += off_1;
            vz_plane_base += off_1;
        }
        vx_curr_plane += off_0;
        vy_curr_plane += off_0;
        vz_curr_plane += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlockSlicePostPred(
    size_t x, DSize_3d size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<T *, 3>& x_diffs,
    std::array<T *, 3>& y_diffs,
    std::array<T *, 3>& z_diffs,
    T *result_start_pos, double errorBound,
    size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    auto idx = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_y = [&](size_t x, size_t y, size_t z)
                 { return x * (size.block_dim2+1) * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_z = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * (size.block_dim3+1) + y * (size.block_dim3+1) + z; };
    const int * vx_y_buffer_pos = buffer_set[0]->currSlice_data_pos;
    const int * vy_y_buffer_pos = buffer_set[1]->currSlice_data_pos;
    const int * vz_y_buffer_pos = buffer_set[2]->currSlice_data_pos;
    T * y_result_pos = result_start_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        const int * vx_z_buffer_pos = vx_y_buffer_pos;
        const int * vy_z_buffer_pos = vy_y_buffer_pos;
        const int * vz_z_buffer_pos = vz_y_buffer_pos;
        T * z_result_pos = y_result_pos;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            const int * vx_top_row_pos = vx_z_buffer_pos - off_0;
            const int * vx_bott_row_pos = vx_z_buffer_pos + off_0;
            const int * vy_prev_row_pos = vy_z_buffer_pos - off_1;
            const int * vy_next_row_pos = vy_z_buffer_pos + off_1;
            const int * vz_curr_row_pos = vz_z_buffer_pos;
            T * result_pos = z_result_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        int dfxx = vx_bott_row_pos[k] - vx_top_row_pos[k];
                        int dfyy = vy_next_row_pos[k] - vy_prev_row_pos[k];
                        int dfzz = vz_curr_row_pos[k+1] - vz_curr_row_pos[k-1];
                        result_pos[k] = (dfxx + dfyy + dfzz) * errorBound;
                    }
                    if(i == 0){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += x_diffs[0][idx(x,y,z)];
                        }
                    }
                    else if(i == size_x-1){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += x_diffs[0][idx(x+1,y,z)];
                        }
                    }
                    if(j == 0){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += y_diffs[1][idx_y(x,y,z)];
                        }
                    }
                    else if(j == size_y-1){
                        for(int k=0; k<size_z; k++){
                            result_pos[k] += y_diffs[1][idx_y(x,y+1,z)];
                        }
                    }
                    result_pos[0] += z_diffs[2][idx_z(x,y,z)];
                    result_pos[size_z-1] += z_diffs[2][idx_z(x,y,z+1)];
                    vz_curr_row_pos += off_1;
                    vy_prev_row_pos += off_1;
                    vy_next_row_pos += off_1;
                    vx_top_row_pos += off_1;
                    vx_bott_row_pos += off_1;
                    result_pos += size.offset_1;
                }
                vz_curr_row_pos += off_0 - size_y * off_1;
                vy_prev_row_pos += off_0 - size_y * off_1;
                vy_next_row_pos += off_0 - size_y * off_1;
                vx_top_row_pos += off_0 - size_y * off_1;
                vx_bott_row_pos += off_0 - size_y * off_1;
                result_pos += size.offset_0 - size_y * size.offset_1;
            }
            vx_z_buffer_pos += size.Bsize;
            vy_z_buffer_pos += size.Bsize;
            vz_z_buffer_pos += size.Bsize;
            z_result_pos += size.Bsize;
        }
        vx_y_buffer_pos += size.Bsize * off_1;
        vy_y_buffer_pos += size.Bsize * off_1;
        vz_y_buffer_pos += size.Bsize * off_1;
        y_result_pos += size.Bsize * size.offset_1;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergence3DProcessBlocksPrePred(
    DSize_3d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set[i]->mean_quant_inds, size.num_blocks);
    }
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0, off_1);
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
            }
            divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
                }
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void divergence3DProcessBlocksPostPred(
    DSize_3d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    std::array<T *, 3>& x_diffs,
    std::array<T *, 3>& y_diffs,
    std::array<T *, 3>& z_diffs,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set[i]->mean_quant_inds, size.num_blocks);
        dxdydz_compute_block_mean_difference(size.block_dim1, size.block_dim2, size.block_dim3, errorBound, cmpkit_set[i]->mean_quant_inds, x_diffs[i], y_diffs[i], z_diffs[i]);
    }
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0, off_1);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
            }
            divergenceProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, result_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                divergenceProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, result_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
                }
                divergenceProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, result_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_divergence(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;

    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<int *, 3> blocks_mean_quant = {nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet_3d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<T *, 3> x_diffs = {nullptr, nullptr, nullptr};
    std::array<T *, 3> y_diffs = {nullptr, nullptr, nullptr};
    std::array<T *, 3> z_diffs = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        blocks_mean_quant[i] = (int *)malloc(size.num_blocks * sizeof(int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], blocks_mean_quant[i]);
        x_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        y_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        z_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            divergence3DProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, x_diffs, y_diffs, z_diffs, divergence_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            divergence3DProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, divergence_pos, errorBound);
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
        free(blocks_mean_quant[i]);
        free(signFlag[i]);
        free(decData[i]);
        free(x_diffs[i]);
        free(y_diffs[i]);
        free(z_diffs[i]);
    }
}

// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize_3d& size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    T *curlx_start_pos, T *curly_start_pos, T *curlz_start_pos, 
    double errorBound, size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    buffer_set[2]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curlx_pos = curlx_start_pos;
    T * curly_pos = curly_start_pos;
    T * curlz_pos = curlz_start_pos;
    const int * vx_curr_plane = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_plane = buffer_set[1]->currSlice_data_pos;
    const int * vz_curr_plane = buffer_set[2]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        // prev & next planes
        const int* vy_plane_prev = vy_curr_plane - off_0;
        const int* vy_plane_next = vy_curr_plane + off_0;
        const int* vz_plane_prev = vz_curr_plane - off_0;
        const int* vz_plane_next = vz_curr_plane + off_0;
        // current planes
        const int* vx_plane_base = vx_curr_plane;
        const int* vy_plane_base = vy_curr_plane;
        const int* vz_plane_base = vz_curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            size_t row_offset = j * off_1;
            // vx-dy
            const int* vx_row_prev = vx_plane_base - off_1;
            const int* vx_row_next = vx_plane_base + off_1;
            // vx-dz
            const int* vx_row = vx_plane_base;

            // vy-dx
            const int* vy_row_prev = vy_plane_prev + row_offset;
            const int* vy_row_next = vy_plane_next + row_offset;
            // vy-dz
            const int* vy_row = vy_plane_base;

            // vz-dx
            const int* vz_row_prev = vz_plane_prev + row_offset;
            const int* vz_row_next = vz_plane_next + row_offset;
            // vz-dy
            const int* vz_row_prev_2 = vz_plane_base - off_1;
            const int* vz_row_next_2 = vz_plane_base + off_1;

            const int* vz_row = vz_plane_base;
            for(size_t k=0; k<size.dim3; k++){
                curlx_pos[index] = ((vz_row_next_2[k] - vz_row_prev_2[k]) - (vy_row[k+1] - vy_row[k-1])) * errorBound;
                curly_pos[index] = ((vx_row[k+1] - vx_row[k-1]) - (vz_row_next[k] - vz_row_prev[k])) * errorBound;
                curlz_pos[index] = ((vy_row_next[k] - vy_row_prev[k]) - (vx_row_next[k] - vx_row_prev[k])) * errorBound;
                index++;
            }
            vx_plane_base += off_1;
            vy_plane_base += off_1;
            vz_plane_base += off_1;
        }
        vx_curr_plane += off_0;
        vy_curr_plane += off_0;
        vz_curr_plane += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlockSlicePostPred(
    size_t x, DSize_3d size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<T *, 3>& x_diffs,
    std::array<T *, 3>& y_diffs,
    std::array<T *, 3>& z_diffs,
    T *curlx_start_pos, T *curly_start_pos, T *curlz_start_pos, 
    double errorBound, size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    for(int h=0; h<3; h++) buffer_set[h]->setGhostEle(size, isTopSlice, isBottomSlice);
    auto idx = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_y = [&](size_t x, size_t y, size_t z)
                 { return x * (size.block_dim2+1) * size.block_dim3 + y * size.block_dim3 + z; };
    auto idx_z = [&](size_t x, size_t y, size_t z)
                 { return x * size.block_dim2 * (size.block_dim3+1) + y * (size.block_dim3+1) + z; };
    const int * vx_y_buffer_pos = buffer_set[0]->currSlice_data_pos;
    const int * vy_y_buffer_pos = buffer_set[1]->currSlice_data_pos;
    const int * vz_y_buffer_pos = buffer_set[2]->currSlice_data_pos;
    T * y_curlx_pos = curlx_start_pos;
    T * y_curly_pos = curly_start_pos;
    T * y_curlz_pos = curlz_start_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        const int * vx_z_buffer_pos = vx_y_buffer_pos;
        const int * vy_z_buffer_pos = vy_y_buffer_pos;
        const int * vz_z_buffer_pos = vz_y_buffer_pos;
        T * z_curlx_pos = y_curlx_pos;
        T * z_curly_pos = y_curly_pos;
        T * z_curlz_pos = y_curlz_pos;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            // vx
            const int * vx_prev_row_pos = vx_z_buffer_pos - off_1;
            const int * vx_next_row_pos = vx_z_buffer_pos + off_1;
            const int * vx_curr_row_pos = vx_z_buffer_pos;
            // vy
            const int * vy_top_row_pos = vy_z_buffer_pos - off_0;
            const int * vy_bott_row_pos = vy_z_buffer_pos + off_0;
            const int * vy_curr_row_pos = vy_z_buffer_pos;
            // vx
            const int * vz_top_row_pos = vz_z_buffer_pos - off_0;
            const int * vz_bott_row_pos = vz_z_buffer_pos + off_0;
            const int * vz_prev_row_pos = vz_z_buffer_pos - off_1;
            const int * vz_next_row_pos = vz_z_buffer_pos + off_1;
            T * curlx_pos = z_curlx_pos;
            T * curly_pos = z_curly_pos;
            T * curlz_pos = z_curlz_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        int xy = vx_next_row_pos[k] - vx_prev_row_pos[k];
                        int xz = vx_curr_row_pos[k+1] - vx_curr_row_pos[k-1];
                        int yx = vy_bott_row_pos[k] - vy_top_row_pos[k];
                        int yz = vy_curr_row_pos[k+1] - vy_curr_row_pos[k-1];
                        int zx = vz_bott_row_pos[k] - vz_top_row_pos[k];
                        int zy = vz_next_row_pos[k] - vz_prev_row_pos[k];
                        curlx_pos[k] = (zy - yz) * errorBound;
                        curly_pos[k] = (xz - zx) * errorBound;
                        curlz_pos[k] = (yx - xy) * errorBound;
                    }
                    if(i == 0){
                        for(int k=0; k<size_z; k++){
                            curly_pos[k] -= x_diffs[2][idx(x,y,z)];
                            curlz_pos[k] += x_diffs[1][idx(x,y,z)];
                        }
                    }
                    else if(i == size_x-1){
                        for(int k=0; k<size_z; k++){
                            curly_pos[k] -= x_diffs[2][idx(x+1,y,z)];
                            curlz_pos[k] += x_diffs[1][idx(x+1,y,z)];
                        }
                    }
                    if(j == 0){
                        for(int k=0; k<size_z; k++){
                            curlx_pos[k] += y_diffs[2][idx_y(x,y,z)];
                            curlz_pos[k] -= y_diffs[0][idx_y(x,y,z)];
                        }
                    }
                    else if(j == size_y-1){
                        for(int k=0; k<size_z; k++){
                            curlx_pos[k] += y_diffs[2][idx_y(x,y+1,z)];
                            curlz_pos[k] -= y_diffs[0][idx_y(x,y+1,z)];
                        }
                    }
                    curlx_pos[0] -= z_diffs[1][idx_z(x,y,z)];
                    curlx_pos[size_z-1] -= z_diffs[1][idx_z(x,y,z+1)];
                    curly_pos[0] += z_diffs[0][idx_z(x,y,z)];
                    curly_pos[size_z-1] += z_diffs[0][idx_z(x,y,z+1)];
                    vx_prev_row_pos += off_1;
                    vx_next_row_pos += off_1;
                    vx_curr_row_pos += off_1;
                    vy_top_row_pos += off_1;
                    vy_bott_row_pos += off_1;
                    vy_curr_row_pos += off_1;
                    vz_top_row_pos += off_1;
                    vz_bott_row_pos += off_1;
                    vz_prev_row_pos += off_1;
                    vz_next_row_pos += off_1;
                    curlx_pos += size.offset_1;
                    curly_pos += size.offset_1;
                    curlz_pos += size.offset_1;
                }
                vx_prev_row_pos += off_0 - size_y * off_1;
                vx_next_row_pos += off_0 - size_y * off_1;
                vx_curr_row_pos += off_0 - size_y * off_1;
                vy_top_row_pos += off_0 - size_y * off_1;
                vy_bott_row_pos += off_0 - size_y * off_1;
                vy_curr_row_pos += off_0 - size_y * off_1;
                vz_top_row_pos += off_0 - size_y * off_1;
                vz_bott_row_pos += off_0 - size_y * off_1;
                vz_prev_row_pos += off_0 - size_y * off_1;
                vz_next_row_pos += off_0 - size_y * off_1;
                curlx_pos += size.offset_0 - size_y * size.offset_1;
                curly_pos += size.offset_0 - size_y * size.offset_1;
                curlz_pos += size.offset_0 - size_y * size.offset_1;
            }
            vx_z_buffer_pos += size.Bsize;
            vy_z_buffer_pos += size.Bsize;
            vz_z_buffer_pos += size.Bsize;
            z_curlx_pos += size.Bsize;
            z_curly_pos += size.Bsize;
            z_curlz_pos += size.Bsize;
        }
        vx_y_buffer_pos += size.Bsize * off_1;
        vy_y_buffer_pos += size.Bsize * off_1;
        vz_y_buffer_pos += size.Bsize * off_1;
        y_curlx_pos += size.Bsize * size.offset_1;
        y_curly_pos += size.Bsize * size.offset_1;
        y_curlz_pos += size.Bsize * size.offset_1;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curl3DProcessBlocksPrePred(
    DSize_3d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    T *curlx_pos, T *curly_pos, T *curlz_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set[i]->mean_quant_inds, size.num_blocks);
    }
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0, off_1);
                recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
            }
            curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PrePred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
                }
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void curl3DProcessBlocksPostPred(
    DSize_3d& size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    std::array<T *, 3>& x_diffs,
    std::array<T *, 3>& y_diffs,
    std::array<T *, 3>& z_diffs,
    T *curlx_pos, T *curly_pos, T *curlz_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++){
        buffer_set[i]->reset();
        extract_block_mean(cmpkit_set[i]->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set[i]->mean_quant_inds, size.num_blocks);
        dxdydz_compute_block_mean_difference(size.block_dim1, size.block_dim2, size.block_dim3, errorBound, cmpkit_set[i]->mean_quant_inds, x_diffs[i], y_diffs[i], z_diffs[i]);
    }
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0, off_1);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
            }
            curlProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0, off_1);
                }
                curlProcessBlockSlicePostPred(x, size, buffer_set, x_diffs, y_diffs, z_diffs, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_curl(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *curlx_result, T *curly_result, T *curlz_result,
    decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;

    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<int *, 3> blocks_mean_quant = {nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet_3d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<T *, 3> x_diffs = {nullptr, nullptr, nullptr};
    std::array<T *, 3> y_diffs = {nullptr, nullptr, nullptr};
    std::array<T *, 3> z_diffs = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        blocks_mean_quant[i] = (int *)malloc(size.num_blocks * sizeof(int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], blocks_mean_quant[i]);
        x_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        y_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        z_diffs[i] = (T *)malloc((size.block_dim1+1)*(size.block_dim2+1)*(size.block_dim3+1)*sizeof(T));
        encode_pos[i] = cmpData[i] + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    }
    T * curlx_pos = curlx_result;
    T * curly_pos = curly_result;
    T * curlz_pos = curlz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            curl3DProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, x_diffs, y_diffs, z_diffs, curlx_pos, curly_pos, curlz_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            curl3DProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, curlx_pos, curly_pos, curlz_pos, errorBound);
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
        free(blocks_mean_quant[i]);
        free(signFlag[i]);
        free(decData[i]);
        free(x_diffs[i]);
        free(y_diffs[i]);
        free(z_diffs[i]);
    }
}

#endif