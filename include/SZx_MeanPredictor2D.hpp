#ifndef _SZX_MEAN_PREDICTOR_2D_HPP
#define _SZX_MEAN_PREDICTOR_2D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZx_app_utils.hpp"
#include "utils.hpp"

template <class T>
void SZx_compress_2dMeanbased(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * block_quant_inds = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
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
            int mean_quant = compute_block_mean_quant(size_x, size_y, size.dim0_offset, y_data_pos, block_buffer_pos, errorBound);
            for(int i=0; i<block_size; i++){
                int err = *block_buffer_pos++ - mean_quant;
                int abs_err = abs(err);
                *sign_pos++ = (err < 0);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
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
void SZx_decompress_2dMeanbased(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                int * pred_err_pos = signPredError;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        *curr_data_pos++ = (*pred_err_pos++ + mean_quant) * 2 * errorBound;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        *curr_data_pos++ = mean_quant * 2 * errorBound;
                    }
                    curr_data_pos += size.dim2 - size_y;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

double SZx_mean_2dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    double mean = compute_mean_2d(size, qmean_pos, errorBound);
    return mean;
}

template <class T>
double SZx_mean_2d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress_2dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZx_mean_2d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZx_mean_2d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            mean = SZx_mean_2dMeanbased(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            exit(0);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

// // No better
// double SZx_variance_2d_postPred_simple_formula(
//     unsigned char *cmpData, size_t dim1, size_t dim2,
//     int blockSideLength, double errorBound
// ){
//     DSize_2d size(dim1, dim2, blockSideLength);
//     unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
//     unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
//     int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
//     unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
//     int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
//     int64_t global_mean = compute_integer_mean_2d<int64_t>(size, qmean_pos, blocks_mean_quant);
//     int block_ind = 0;
//     int64_t integer_sum = 0;
//     for(size_t x=0; x<size.block_dim1; x++){
//         int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
//         for(size_t y=0; y<size.block_dim2; y++){
//             int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
//             int block_size = size_x * size_y;
//             int fixed_rate = (int)cmpData[block_ind];
//             int block_mean = blocks_mean_quant[block_ind++];
//             int mean_err = block_mean - global_mean;
//             int block_err_sum = 0;
//             if(fixed_rate){
//                 size_t cmp_block_sign_length = (block_size + 7) / 8;
//                 convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
//                 encode_pos += cmp_block_sign_length;
//                 unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
//                 encode_pos += savedbitsbytelength;
//                 convert2SignIntArray(signFlag, signPredError, block_size);
//                 for(int i=0; i<block_size; i++){
//                     integer_sum += signPredError[i] * signPredError[i];
//                     block_err_sum += signPredError[i];
//                 }
//                 integer_sum += block_err_sum * mean_err * 2 + mean_err * mean_err * block_size;
//             }else{
//                 integer_sum += mean_err * mean_err * block_size;
//             }
//         }
//     }
//     free(signPredError);
//     free(signFlag);
//     free(blocks_mean_quant);
//     double var = (double)integer_sum / (size.nbEle - 1) * (2 * errorBound) * (2 * errorBound);
//     return var;
// }
double SZx_variance_2d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_2d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    int64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
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
                    int diff = signPredError[i] + mean_err;
                    squared_sum += diff * diff;
                }
            }else{
                squared_sum += mean_err * mean_err * block_size;
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double var = (2 * errorBound) * (2 * errorBound) * (double)squared_sum / (size.nbEle - 1);
    return var;
}

double SZx_variance_2d_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    extract_block_mean(qmean_pos, blocks_mean_quant, size.num_blocks);
    int block_ind = 0;
    int64_t quant_sum = 0, squared_quant_sum = 0;
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                for(int i=0; i<block_size; i++){
                    int curr_quant = signPredError[i] + block_mean;
                    quant_sum += curr_quant;
                    squared_quant_sum += curr_quant * curr_quant;
                }
            }else{
                quant_sum += block_mean * block_size;
                squared_quant_sum += block_mean * block_mean * block_size;
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
double SZx_variance_2d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress_2dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZx_variance_2d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZx_variance_2d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZx_variance_2d_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            var = SZx_variance_2d_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockRow2PrePred(
    size_t x, DSize_2d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t buffer_dim0_offset
){
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
                curr_buffer_pos += buffer_dim0_offset;
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            int * data_pos = cmpkit_set->signPredError;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = data_pos[j] + mean_quant;
                }
                curr_buffer_pos += buffer_dim0_offset;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
}

inline void recoverBlockRow2PostPred(
    size_t x, DSize_2d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t buffer_dim0_offset
){
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
                memset(curr_buffer_pos+i*buffer_dim0_offset, 0, size_y*sizeof(int));
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            int * data_pos = cmpkit_set->signPredError;
            for(int i=0; i<size_x; i++){
                memcpy(curr_buffer_pos+i*buffer_dim0_offset, data_pos+i*size_y, size_y*sizeof(int));
            }
        }        
        buffer_start_pos += size.Bsize;
    }
}

template <class T>
inline void dxdyProcessBlockRowPrePred(
    size_t x, DSize_2d size, SZxAppBufferSet_2d *buffer_set,
    SZxCmpBufferSet *cmpkit_set, T *dx_start_pos, T *dy_start_pos,
    double errorBound, bool isTopRow, bool isBottomRow
){
    int block_ind_offset = x * size.block_dim2;
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int * prevBlockBottom_pos = buffer_set->prevRow_data_pos + (size.Bsize - 1) * size.dim2;
    int * nextBlockTop_pos = buffer_set->nextRow_data_pos;
    int * curr_row = buffer_set->currRow_data_pos;
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    for(int i=0; i <size_x; i++){
        int * prev_row_pos = i > 0 ? curr_row - buffer_set->buffer_dim0_offset
                             : isTopRow ? curr_row : prevBlockBottom_pos;
        int * next_row_pos = i < size_x - 1 ? curr_row + buffer_set->buffer_dim0_offset
                             : isBottomRow ? curr_row : nextBlockTop_pos;
        bool isBlockTop = i == 0;
        bool isBlockBottom = i == size_x - 1;
        bool isTopEle = isTopRow && isBlockTop;
        bool isBottomEle = isBottomRow && isBlockBottom;
        int coeff_dx = isTopEle || isBottomEle ? 2 : 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_ind = block_ind_offset + y;
            bool isHeadBlock = y == 0;
            bool isTailBlock = y == size.block_dim2 - 1;
            for(int j=0; j<size_y; j++){
                bool isBlockHeadEle = j == 0;
                bool isBlockTailEle = j == size_y - 1;
                bool isHeadEle = isHeadBlock && isBlockHeadEle;
                bool isTailEle = isTailBlock && isBlockTailEle;
                int coeff_dy = isHeadEle || isTailEle ? 2 : 1;
                size_t curr_ind = y * size.Bsize + j;
                size_t prev_ind = isHeadEle ? curr_ind : curr_ind - 1;
                size_t next_ind = isTailEle ? curr_ind : curr_ind + 1;
                int dx_integer = next_row_pos[curr_ind] - prev_row_pos[curr_ind];
                int dy_integer = curr_row[next_ind] - curr_row[prev_ind];
                dx_pos[curr_ind] = dx_integer * coeff_dx * errorBound;
                dy_pos[curr_ind] = dy_integer * coeff_dy * errorBound;
            }
        }
        dx_pos += size.dim2;
        dy_pos += size.dim2;
        curr_row += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdyProcessBlockRowPostPred(
    size_t x, DSize_2d size, SZxAppBufferSet_2d *buffer_set,
    SZxCmpBufferSet *cmpkit_set, T *dx_start_pos, T *dy_start_pos,
    double errorBound, bool isTopRow, bool isBottomRow
){
    int block_ind_offset = x * size.block_dim2;
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int * prevBlockBottom_pos = buffer_set->prevRow_data_pos + (size.Bsize - 1) * size.dim2;
    int * nextBlockTop_pos = buffer_set->nextRow_data_pos;
    int * curr_row = buffer_set->currRow_data_pos;
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    for(int i=0; i <size_x; i++){
        int * prev_row_pos = i > 0 ? curr_row - buffer_set->buffer_dim0_offset
                             : isTopRow ? curr_row : prevBlockBottom_pos;
        int * next_row_pos = i < size_x - 1 ? curr_row + buffer_set->buffer_dim0_offset
                             : isBottomRow ? curr_row : nextBlockTop_pos;
        bool isBlockTop = i == 0;
        bool isBlockBottom = i == size_x - 1;
        bool isTopEle = isTopRow && isBlockTop;
        bool isBottomEle = isBottomRow && isBlockBottom;
        bool dx_flag = (isBlockTop && !isTopRow) || (isBlockBottom && !isBottomRow);
        int coeff_dx = isTopEle || isBottomEle ? 2 : 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_ind = block_ind_offset + y;
            bool isHeadBlock = y == 0;
            bool isTailBlock = y == size.block_dim2 - 1;
            for(int j=0; j<size_y; j++){
                bool isBlockHeadEle = j == 0;
                bool isBlockTailEle = j == size_y - 1;
                bool isHeadEle = isHeadBlock && isBlockHeadEle;
                bool isTailEle = isTailBlock && isBlockTailEle;
                bool dy_flag = (isBlockHeadEle && !isHeadBlock) || (isBlockTailEle && !isTailBlock);
                int coeff_dy = isHeadEle || isTailEle ? 2 : 1;
                size_t curr_ind = y * size.Bsize + j;
                size_t prev_ind = isHeadEle ? curr_ind : curr_ind - 1;
                size_t next_ind = isTailEle ? curr_ind : curr_ind + 1;
                int dx_integer = next_row_pos[curr_ind] - prev_row_pos[curr_ind];
                int dy_integer = curr_row[next_ind] - curr_row[prev_ind];
                {
                    if(dy_flag){
                        int dy_delta = isBlockHeadEle ? cmpkit_set->mean_quant_inds[block_ind] - cmpkit_set->mean_quant_inds[block_ind-1]
                                    : cmpkit_set->mean_quant_inds[block_ind+1] - cmpkit_set->mean_quant_inds[block_ind];
                        dy_integer += dy_delta;
                    }
                    if(dx_flag){
                        int dx_delta = isBlockTop ? cmpkit_set->mean_quant_inds[block_ind] - cmpkit_set->mean_quant_inds[block_ind-size.block_dim2]
                                    : cmpkit_set->mean_quant_inds[block_ind+size.block_dim2] - cmpkit_set->mean_quant_inds[block_ind];
                        dx_integer += dx_delta;
                    }
                }
                dx_pos[curr_ind] = dx_integer * coeff_dx * errorBound;
                dy_pos[curr_ind] = dy_integer * coeff_dy * errorBound;
            }
        }
        dx_pos += size.dim2;
        dy_pos += size.dim2;
        curr_row += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_2d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    int * tempBlockRow = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPrePred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempBlockRow);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockRowPrePred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                dxdyProcessBlockRowPrePred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
}

template <class T>
inline void dxdyProcessBlocksPostPred(
    DSize_2d& size,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    int * tempBlockRow = nullptr;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PostPred(x, size, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempBlockRow);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockRowPostPred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PostPred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                dxdyProcessBlockRowPostPred(x, size, buffer_set, cmpkit_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
}

template <class T>
void SZx_dxdy_2dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int * Buffer_2d = (int *)malloc(size.Bsize * size.dim2 * 3 * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    SZxAppBufferSet_2d * buffer_set = new SZxAppBufferSet_2d(size.Bsize, size.dim2, Buffer_2d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdyProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            SZx_decompress_2dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            compute_dxdy(dim1, dim2, decData, dx_pos, dy_pos);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(Buffer_2d);
    free(decData);
}

// heatdis recover
inline void recoverBlockRow2PrePred(
    size_t x, DSize_2d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos, int current,
    size_t buffer_dim0_offset
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    int block_ind_offset = x * size.block_dim2;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_ind = block_ind_offset + y;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpkit_set->cmpData[current][block_ind];
        int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = mean_quant;
                }
                curr_buffer_pos += buffer_dim0_offset;
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            int * data_pos = cmpkit_set->signPredError;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = data_pos[j] + mean_quant;
                }
                curr_buffer_pos += buffer_dim0_offset;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
}

inline void heatdisProcessCompressBlockRowPrePred(
    size_t x, DSize_2d size, Temperature_info temp_info,
    SZxCmpBufferSet *cmpkit_set, SZxAppBufferSet_2d *buffer_set,
    double errorBound, int iter, int next,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int block_ind_offset = x * size.block_dim2;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_prepred(buffer_set->currRow_data_pos, size, size_x, buffer_set->buffer_dim0_offset, temp_info, isTopRow, isBottomRow);
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + INT_BYTES * block_ind_offset;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = encode_pos;
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_ind = block_ind_offset + y;
        int block_size = size_x * size_y;
        const int * block_buffer_pos = buffer_start_pos;
        int pred = heatdis_update_block_mean(block_buffer_pos-buffer_set->buffer_dim0_offset-1, buffer_set->buffer_dim0_offset, size_x, size_y);
        unsigned char * sign_pos = cmpkit_set->signFlag;
        unsigned int * abs_err_pos = cmpkit_set->absPredError;
        int abs_err, max_err = 0;
        for(int i=0; i<size_x; i++){
            const int * curr_buffer_pos = block_buffer_pos;
            for(int j=0; j<size_y; j++){
                int quant = integerize_quant(curr_buffer_pos++, buffer_set->buffer_dim0_offset);
                int err = quant - pred;
                *sign_pos++ = (err < 0);
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            block_buffer_pos += buffer_set->buffer_dim0_offset;
        }
        buffer_start_pos += size_y;
        int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
        cmpData[block_ind] = (unsigned char)fixed_rate;
        for(int k=3; k>=0; k--){
            *(qmean_pos++) = (pred >> (8 * k)) & 0xff;
        }
        if(fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, encode_pos);
            encode_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, encode_pos, fixed_rate);
            encode_pos += savedbitsbyteLength;
        }
    }
    size_t increment = encode_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void heatdisUpdatePrePred(
    DSize_2d size, SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_2d *buffer_set, Temperature_info temp_info,
    double errorBound, int current, int next, int iter
){
    size_t buffer_dim0_offset = buffer_set->buffer_dim0_offset;
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, current, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, current, buffer_set->buffer_dim0_offset);
            heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, current, buffer_set->buffer_dim0_offset);
                heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, false, false);
            }
        }
    }
}

inline void heatdisUpdatePrePred(
    DSize_2d& size,
    SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_2d *buffer_set,
    Temperature_info& temp_info,
    double errorBound,
    int max_iter
){
    int current = 0, next = 1;
    for(int iter=0; iter<max_iter; iter++){
        heatdisUpdatePrePred(size, cmpkit_set, buffer_set, temp_info, errorBound, current, next, iter);
        current = next;
        next = 1 - current;
    }
}

template <class T>
void SZx_heatdis_2dMeanbased(
    unsigned char *compressed_data,
    size_t dim1, size_t dim2, int blockSideLength, int max_iter,
    float source_temp, float wall_temp, float init_temp, float ratio,
    double errorBound, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * signPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        cmpData[i] = (unsigned char *)malloc(size.nbEle * sizeof(T)*2);
        offsets[i] = (int *)malloc(size.block_dim1 * sizeof(int));
    }
    memcpy(cmpData[0], compressed_data, size.nbEle * sizeof(T));
    int current = 0, next = 1;
    size_t prefix_length = 0;
    int block_index = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        offsets[0][x] = prefix_length;
        offsets[1][x] = 0;
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1) * size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y * size.Bsize;
            int block_size = size_x * size_y;
            int cmp_block_sign_length = (block_size + 7) / 8;
            int fixed_rate = (int)compressed_data[block_index];
            block_index++;
            size_t savedbitsbytelength = compute_encoding_byteLength(block_size, fixed_rate);
            if(fixed_rate)
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
    }
    Temperature_info temp_info(source_temp, wall_temp, init_temp, ratio, errorBound);
    SZxAppBufferSet_2d * buffer_set = new SZxAppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d, appType::HEATDIS);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, offsets, blocks_mean_quant, absPredError, signPredError, signFlag);
    int status = max_iter % 2;
    size_t cmpSize = 0;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            unsigned char * compressed = cmpData[status];
            for(int i=0; i<max_iter; i++){
                SZx_decompress_2dMeanbased(decData, compressed, dim1, dim2, blockSideLength, errorBound);
                doWork(dim1, dim2, decData, h, source_temp, wall_temp, ratio);
                SZx_compress_2dMeanbased(decData, compressed, dim1, dim2, blockSideLength, errorBound, cmpSize);
            }
            break;
        }
        case decmpState::prePred:{
            heatdisUpdatePrePred(size, cmpkit_set, buffer_set, temp_info, errorBound, max_iter);
            cmpSize = size.num_blocks + cmpkit_set->cmpSize;
            break;
        }
        case decmpState::postPred:{
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
    free(signPredError);
    free(signFlag);
    free(decData);
    free(h);
    free(blocks_mean_quant);
    for(int i=0; i<2; i++){
        free(cmpData[i]);
        free(offsets[i]);
    }
    free(cmpData);
    free(offsets);
}

#endif