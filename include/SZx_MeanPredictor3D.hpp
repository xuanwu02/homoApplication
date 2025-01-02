#ifndef _SZX_MEAN_BASED_2D_HPP
#define _SZX_MEAN_BASED_2D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZx_app_utils.hpp"

template <class T>
void SZx_compress_3dMeanbased(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
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
                int mean_quant = compute_block_mean_quant(size_x, size_y, size_z, size.dim0_offset, size.dim1_offset, z_data_pos, block_quant_inds, errorBound);
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

double SZx_mean_3dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    double mean = compute_mean_3d(size, qmean_pos, errorBound);
    return mean;
}

double SZx_variance_3dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int * block_quant_inds = (int *)malloc(size.max_num_block_elements * sizeof(int));
    int64_t global_mean = compute_integer_mean_3d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    int64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
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
                        int diff = signPredError[i] + block_mean - global_mean;
                        squared_sum += diff * diff;
                    }
                }else{
                    squared_sum += (block_mean - global_mean) * (block_mean - global_mean) * block_size;
                }
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(block_quant_inds);
    double var = (2 * errorBound) * (2 * errorBound) * squared_sum / (size.nbEle - 1);
    return var;
}

inline void recoverBlockPlane2PrePred(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
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
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred(
    size_t x, DSize_3d size, SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_3d *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopPlane, bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    size_t buffer_index_offset = buffer_dim1_offset + 1;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_dim0_offset - buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_dim1_offset - 1;
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * dx_pos = dx_start_pos + i * size.dim0_offset;
        T * dy_pos = dy_start_pos + i * size.dim0_offset;
        T * dz_pos = dz_start_pos + i * size.dim0_offset;
        const int * prev_plane = i > 0 ? curr_plane - buffer_dim0_offset
                               : isTopPlane ? curr_plane : prevBlockPlaneBottom_pos;
        const int * next_plane = i < size_x - 1 ? curr_plane + buffer_dim0_offset
                               : isBottomPlane ? curr_plane : nextBlockPlaneTop_pos;
        const int * curr_row = curr_plane + buffer_dim1_offset + 1;
        int coeff_dx = (isTopPlane && i == 0) || (isBottomPlane && i == size_x - 1) ? 2 : 1;
        for(size_t j=0; j<size.dim2; j++){
            const int * prev_row = j == 0 ? curr_row : curr_row - buffer_dim1_offset;
            const int * next_row = j == size.dim2 - 1 ? curr_row : curr_row + buffer_dim1_offset;
            int coeff_dy = (j == 0) || (j == size.dim2 - 1) ? 2 : 1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index = (j + 1) * buffer_dim1_offset + k + 1;
                size_t res_index = j * size.dim1_offset + k;
                size_t prev_k = k == 0 ? k : k - 1;
                size_t next_k = k == size.dim3 - 1 ? k : k + 1;
                int coeff_dz = (k == 0) || (k == size.dim3 - 1) ? 2 : 1;
                dx_pos[res_index] = (next_plane[buffer_index] - prev_plane[buffer_index]) * coeff_dx * errorBound;
                dy_pos[res_index] = (next_row[k] - prev_row[k]) * coeff_dy * errorBound;
                dz_pos[res_index] = (curr_row[next_k] - curr_row[prev_k]) * coeff_dz * errorBound;
            }
            curr_row += buffer_dim1_offset;
        }
        curr_plane += buffer_dim0_offset;
    }
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_3d *buffer_set,
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
            recoverBlockPlane2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            recoverBlockPlane2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
            dxdydzProcessBlockPlanePrePred(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockPlanePrePred(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockPlane2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextPlane_data_pos, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                dxdydzProcessBlockPlanePrePred(x, size, cmpkit_set, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
}

template <class T>
void SZx_dxdydz_3dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound, T *dx_result,
    T *dy_result, T *dz_result, decmpState state
){
    const DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 1;
    size_t buffer_dim2 = size.dim2 + 1;
    size_t buffer_dim3 = size.dim3 + 1;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    SZxAppBufferSet_3d * buffer_set = new SZxAppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;
    switch(state){
        case decmpState::postPred:{
            printf("Not implemented\n");
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
        }
    }
    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_3d);
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

#endif