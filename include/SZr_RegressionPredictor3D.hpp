#ifndef _SZR_REG_PREDICTOR_3D_HPP
#define _SZR_REG_PREDICTOR_3D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZr_app_utils.hpp"

template <class T>
void SZr_compress_3dRegression(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_3D * sizeof(float));
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_3D * FLOAT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        const T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            const T * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate, max_err = 0;
                unsigned int * abs_err_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                const T * curr_data_pos = z_data_pos;
                compute_regression_coeffcients_3d(z_data_pos, size_x, size_y, size_z, size.dim0_offset, size.dim1_offset, reg_coeff);
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        for(int k=0; k<size_z; k++){
                            T pred = predict_regression_3d<T>(i, j, k, reg_coeff);
                            int err = SZ_quantize(curr_data_pos[k] - pred, errorBound);
                            int abs_err = abs(err);
                            *sign_pos++ = (err < 0);
                            *abs_err_pos++ = abs_err;
                            max_err = max_err > abs_err ? max_err : abs_err;
                        }
                        curr_data_pos += size.dim1_offset;
                    }
                    curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                }
                fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                save_regression_coeff_3d(reg_coeff_pos, reg_coeff);
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
    free(reg_coeff);
}

template <class T>
void SZr_decompress_3dRegression(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_3D * sizeof(float));
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_3D * FLOAT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            T * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                T * curr_data_pos = z_data_pos;
                extract_regression_coeff_3d(reg_coeff_pos, reg_coeff);
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
                                T pred = predict_regression_3d<T>(i, j, k, reg_coeff);
                                curr_data_pos[k] = pred + *pred_err_pos++ * 2 * errorBound;
                            }
                            curr_data_pos += size.dim1_offset;
                        }
                        curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                    }
                }else{
                    T pred = reg_coeff[REG_COEFF_SIZE_3D-1];
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                curr_data_pos[j] = pred;
                            }
                            curr_data_pos += size.dim1_offset;
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
    free(reg_coeff);
}

double SZr_mean_3dRegression(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_3D * sizeof(float));
    double mean = compute_mean_3d(size, reg_coeff_pos, reg_coeff);
    free(reg_coeff);
    return mean;
}

template <class T>
double SZr_variance_3dRegression(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_3D * FLOAT_BYTES) * size.num_blocks;
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_3D * sizeof(float));
    double mean = compute_mean_3d(size, reg_coeff_pos, reg_coeff);
    double var = 0;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                extract_regression_coeff_3d(reg_coeff_pos, reg_coeff);
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
                                T pred = predict_regression_3d<T>(i, j, k, reg_coeff);
                                T curr_data = pred + (*pred_err_pos++) * 2 * errorBound;
                                var += (curr_data - mean) * (curr_data - mean);
                            }
                        }
                    }
                }else{
                    T pred = reg_coeff[REG_COEFF_SIZE_3D-1];
                    var += (pred - mean) * (pred - mean) * block_size;
                }
            }
        }
    }
    var /= (size.nbEle - 1);
    return var;
}

#endif