#ifndef _SZX_MEAN_PREDICTOR_3D_HPP
#define _SZX_MEAN_PREDICTOR_3D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZx_app_utils.hpp"
#include "utils.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZx_compress_3dMeanbased(
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
                int mean_quant = compute_block_mean_quant(size_x, size_y, size_z, size.dim0_offset, size.dim1_offset, z_data_pos, block_quant_inds, inver_eb);
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
                for(int k=3; k>=0; k--){
                    *(qmean_pos++) = (mean_quant >> (8 * k)) & 0xff;
                }
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                    encode_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                    encode_pos += savedbitsbyteLength;
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(block_quant_inds);
}

template <class T>
void SZx_decompress_3dMeanbased(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * pred_err_pos = signPredError;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = (*pred_err_pos++ + mean_quant) * 2 * errorBound;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = mean_quant * 2 * errorBound;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_to_PrePred_3dMeanbased(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * pred_err_pos = signPredError;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = *pred_err_pos++ + mean_quant;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = mean_quant;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

void SZx_decompress_to_PostPred_3dMeanbased(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * pred_err_pos = signPredError;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = *pred_err_pos++;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }else{
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                *curr_data_pos++ = 0;
                            }
                            curr_data_pos += size.dim1_offset - size_z;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

template <class T>
double SZx_mean_3d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

double SZx_mean_3d_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * pred_err_pos = signPredError;
                    for(int i=0; i<block_size; i++){
                        quant_sum += *pred_err_pos++ + mean_quant;
                    }
                }else{
                    quant_sum += mean_quant * block_size;
                }
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

// double SZx_mean_3d_postPred2(
//     unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
//     int blockSideLength, double errorBound
// ){
//     DSize_3d size(dim1, dim2, dim3, blockSideLength);
//     unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
//     double mean = compute_mean_3d(size, qmean_pos, errorBound);
//     return mean;
// }
double SZx_mean_3d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    for(int i=0; i<block_size; i++){
                        quant_sum += signPredError[i];
                    }
                }
                quant_sum += mean_quant * block_size;
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZx_mean_3d(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZx_mean_3d_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            mean = SZx_mean_3d_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_mean_3d_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_variance_3d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_3d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    uint64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    for(int i=0; i<block_size; i++){
                        int64_t d = static_cast<int64_t>(signPredError[i] + mean_err);
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
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double var = (2 * errorBound) * (2 * errorBound) * (double)squared_sum / (size.nbEle - 1);
    return var;
}

double SZx_variance_3d_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                uint64_t block_size = size_x * size_y * size_z;
                int block_mean = blocks_mean_quant[block_ind];
                int fixed_rate = (int)cmpData[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    for(int i=0; i<block_size; i++){
                        int curr_quant = signPredError[i] + block_mean;
                        quant_sum += curr_quant;
                        squared_quant_sum += (uint64_t)curr_quant * curr_quant;
                    }
                }else{
                    quant_sum += block_mean * block_size;
                    squared_quant_sum += block_mean * block_mean * block_size;
                }
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double var = ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1) * (2 * errorBound) * (2 * errorBound);
    return var;
}

template <class T>
double SZx_variance_3d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZx_variance_3d(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZx_variance_3d_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZx_variance_3d_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            var = SZx_variance_3d_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockPlane2PrePred_IntBuffer(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                int * data_pos = cmpkit_set->signPredError;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] = data_pos[k] + mean_quant;
                        }
                        curr_buffer_pos += buffer_dim1_offset;
                        data_pos += size_z;
                    }
                    curr_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] =  mean_quant;
                        }
                        curr_buffer_pos += buffer_dim1_offset;
                    }
                    curr_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void recoverBlockPlane2PrePred_FltBuffer(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, T *buffer_data_pos, double errorBound,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    T * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        for(size_t z=0; z<size.block_dim3; z++){
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            T * curr_buffer_pos = buffer_start_pos;
            int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                int * data_pos = cmpkit_set->signPredError;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] = (data_pos[k] + mean_quant) * errorBound;
                        }
                        curr_buffer_pos += buffer_dim1_offset;
                        data_pos += size_z;
                    }
                    curr_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] =  mean_quant * errorBound;
                        }
                        curr_buffer_pos += buffer_dim1_offset;
                    }
                    curr_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred_IntBuffer(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_3d<int> *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.dim0_offset;
        T * x_dy_pos = dy_start_pos + i * size.dim0_offset;
        T * x_dz_pos = dz_start_pos + i * size.dim0_offset;
        const int * prev_plane = curr_plane - buffer_set->buffer_dim0_offset;
        const int * next_plane = curr_plane + buffer_set->buffer_dim0_offset;
        const int * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.dim1_offset;
            T * y_dy_pos = x_dy_pos + j * size.dim1_offset;
            T * y_dz_pos = x_dz_pos + j * size.dim1_offset;
            const int * prev_row = curr_row - buffer_set->buffer_dim1_offset;
            const int * next_row = curr_row + buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_dx_pos[index_1d] = (next_plane[buffer_index_2d] - prev_plane[buffer_index_2d]) * errorBound;
                y_dy_pos[index_1d] = (next_row[index_1d] - prev_row[index_1d]) * errorBound;
                y_dz_pos[index_1d] = (curr_row[index_1d + 1] - curr_row[index_1d - 1]) * errorBound;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred_FltBuffer(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_3d<T> *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopPlane, bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const T * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const T * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    const T * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.dim0_offset;
        T * x_dy_pos = dy_start_pos + i * size.dim0_offset;
        T * x_dz_pos = dz_start_pos + i * size.dim0_offset;
        const T * prev_plane = curr_plane - buffer_set->buffer_dim0_offset;
        const T * next_plane = curr_plane + buffer_set->buffer_dim0_offset;
        const T * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.dim1_offset;
            T * y_dy_pos = x_dy_pos + j * size.dim1_offset;
            T * y_dz_pos = x_dz_pos + j * size.dim1_offset;
            const T * prev_row = curr_row - buffer_set->buffer_dim1_offset;
            const T * next_row = curr_row + buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_dx_pos[index_1d] = next_plane[buffer_index_2d] - prev_plane[buffer_index_2d];
                y_dy_pos[index_1d] = next_row[index_1d] - prev_row[index_1d];
                y_dz_pos[index_1d] = curr_row[index_1d + 1] - curr_row[index_1d - 1];
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdydzProcessBlocksPrePred_IntBuffer(
    DSize_3d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_3d<int> *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockPlane = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred_IntBuffer(x, size, cmpkit_set, encode_pos, buffer_set->currPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            recoverBlockPlane2PrePred_IntBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            dxdydzProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockPlane2PrePred_IntBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                dxdydzProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdydzProcessBlocksPrePred_FltBuffer(
    DSize_3d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_3d<T> *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    T * tempBlockPlane = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred_FltBuffer(x, size, cmpkit_set, encode_pos, buffer_set->currPlane_data_pos, errorBound, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            recoverBlockPlane2PrePred_FltBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, errorBound, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            dxdydzProcessBlockPlanePrePred_FltBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockPlanePrePred_FltBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockPlane2PrePred_FltBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, errorBound, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                dxdydzProcessBlockPlanePrePred_FltBuffer(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_dxdydz_3dMeanbased_IntBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound, T *dx_result,
    T *dy_result, T *dz_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    SZxAppBufferSet_3d<int> * buffer_set = new SZxAppBufferSet_3d<int>(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            // printf("Not implemented\n");
            printf("recover_time = %.6f\n", rec_time);
            printf("process_time = %.6f\n", op_time);
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(decData);
    free(blocks_mean_quant);
}

template <class T>
void SZx_dxdydz_3dMeanbased_FltBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound, T *dx_result,
    T *dy_result, T *dz_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    T * Buffer_3d = (T *)malloc(buffer_size * 3 * sizeof(T));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    SZxAppBufferSet_3d<T> * buffer_set = new SZxAppBufferSet_3d<T>(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            // printf("Not implemented\n");
            printf("recover_time = %.6f\n", rec_time);
            printf("process_time = %.6f\n", op_time);
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred_FltBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(decData);
    free(blocks_mean_quant);
}

template <class T>
inline void laplacianProcessBlockPlanePrePred_IntBuffer(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_3d<int> *buffer_set, double errorBound,
    T *result_start_pos, bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * x_result_pos = result_start_pos + i * size.dim0_offset;
        const int * prev_plane = curr_plane - buffer_set->buffer_dim0_offset;
        const int * next_plane = curr_plane + buffer_set->buffer_dim0_offset;
        const int * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_result_pos = x_result_pos + j * size.dim1_offset;
            const int * prev_row = curr_row - buffer_set->buffer_dim1_offset;
            const int * next_row = curr_row + buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_result_pos[k] = (curr_row[k - 1] + curr_row[k + 1] +
                                   prev_row[k] + next_row[k] +
                                   prev_plane[buffer_index_2d] + next_plane[buffer_index_2d] -
                                   6 * curr_row[k]) * errorBound * 2;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred_IntBuffer(
    DSize_3d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_3d<int> *buffer_set,
    unsigned char *encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    int * tempBlockPlane = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred_IntBuffer(x, size, cmpkit_set, encode_pos, buffer_set->currPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            recoverBlockPlane2PrePred_IntBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            laplacianProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, false, true);
            }else{
                recoverBlockPlane2PrePred_IntBuffer(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                laplacianProcessBlockPlanePrePred_IntBuffer(x, size, cmpkit_set, buffer_set, errorBound, result_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian_3dMeanbased_IntBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    SZxAppBufferSet_3d<int> * buffer_set = new SZxAppBufferSet_3d<int>(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            // printf("Not implemented\n");
            printf("recover_time = %.6f\n", rec_time);
            printf("process_time = %.6f\n", op_time);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress_3dMeanbased(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(decData);
    free(blocks_mean_quant);
}

#endif