#ifndef _SZX_MEAN_PREDICTOR_2D_1D_HPP
#define _SZX_MEAN_PREDICTOR_2D_1D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZx_app_utils.hpp"
#include "utils.hpp"
#include "settings.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZx_compress2D_1dMeanbased(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    const DSize_2d1d size(dim1, dim2, blockSideLength);
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
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate, max_err = 0;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            int * block_buffer_pos = block_quant_inds;
            int mean_quant = compute_block_mean_quant(block_size, y_data_pos, block_buffer_pos, inver_eb);
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
        x_data_pos += size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(block_quant_inds);
}

template <class T>
void SZx_decompress2D_1dMeanbased(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d1d size(dim1, dim2, blockSideLength);
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
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
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
                for(int i=0; i<block_size; i++){
                    *curr_data_pos++ = (*pred_err_pos++ + mean_quant) * 2 * errorBound;
                }
            }else{
                for(int i=0; i<block_size; i++){
                    *curr_data_pos++ = mean_quant * 2 * errorBound;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

// double SZx_mean2D_1d_postPred2(
//     unsigned char *cmpData, size_t dim1, size_t dim2,
//     int blockSideLength, double errorBound
// ){
//     DSize_2d1d size(dim1, dim2, blockSideLength);
//     unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
//     ino64_t sum = 0;
//     for(size_t x=0; x<size.block_dim1; x++){
//         for(size_t y=0; y<size.block_dim2; y++){
//             int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
//             int mean = (0xff000000 & (*qmean_pos << 24)) |
//                         (0x00ff0000 & (*(qmean_pos+1) << 16)) |
//                         (0x0000ff00 & (*(qmean_pos+2) << 8)) |
//                         (0x000000ff & *(qmean_pos+3));
//             qmean_pos += 4;
//             sum += mean * block_size;
//         }
//     }
//     double mean = sum * 2 * errorBound / size.nbEle;
//     return mean;
// }

double SZx_mean2D_1d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_2d_1d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
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
                    quant_sum += signPredError[i];
                }
            }
            quant_sum += block_mean * block_size;
        }
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double mean = (2 * errorBound) * (double)quant_sum / size.nbEle;
    return mean;
}

template <class T>
double SZx_mean2D_1d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress2D_1dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZx_mean2D_1d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean = -9999;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZx_mean2D_1d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            // mean = SZx_mean_2d_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            mean = SZx_mean2D_1d_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZx_variance2D_1d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    int64_t global_mean = compute_integer_mean_2d_1d<int64_t>(size, qmean_pos, blocks_mean_quant);
    int block_ind = 0;
    uint64_t squared_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            uint64_t block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
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
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    double var = (2 * errorBound) * (2 * errorBound) * (double)squared_sum / (size.nbEle - 1);
    return var;
}

template <class T>
double SZx_variance2D_1d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZx_decompress2D_1dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZx_variance2D_1d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var = -9999;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZx_variance2D_1d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            break;
        }
        case decmpState::postPred:{
            var = SZx_variance2D_1d_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockRow2PrePred(
    size_t x, DSize_2d1d size, unsigned char *cmpData,
    SZxCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int block_ind = x * size.Bwidth * size.block_dim2;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int * curr_buffer_pos = buffer_start_pos;
            int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
            int fixed_rate = (int)cmpData[block_ind];
            block_ind++;
            if(!fixed_rate){
                for(int j=0; j<block_size; j++){
                    curr_buffer_pos[j] = mean_quant;
                }
            }
            else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                for(int j=0; j<block_size; j++){
                    curr_buffer_pos[j] = cmpkit_set->signPredError[j] + mean_quant;
                }
            }        
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockRowPrePred(
    size_t x, DSize_2d1d& size, SZxAppBufferSet_2d1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    const int * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const int * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            *dx_pos++ = (next_row[j] - prev_row[j]) * errorBound;
            *dy_pos++ = (curr_row[j+1] - curr_row[j-1]) * errorBound;
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_2d1d& size,
    size_t numBlockRow,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_2d1d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bwidth * size.dim2;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<numBlockRow; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numBlockRow - 1){
                dxdyProcessBlockRowPrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                dxdyProcessBlockRowPrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_dxdy_1dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    size_t numblockRow = (size.dim1 - 1) / size.Bwidth + 1;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZxAppBufferSet_2d1d * buffer_set = new SZxAppBufferSet_2d1d(buffer_dim1, buffer_dim2, Buffer_2d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
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
            dxdyProcessBlocksPrePred(size, numblockRow, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress2D_1dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockRowPrePred(
    size_t x, DSize_2d1d size, SZxAppBufferSet_2d1d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x * size.Bwidth;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * laplacian_pos = result_start_pos;
    const int * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const int * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            *laplacian_pos++ = (curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j]) * errorBound * 2;
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacian2DProcessBlocksPrePred(
    DSize_2d1d& size,
    size_t numBlockRow,
    SZxCmpBufferSet *cmpkit_set, 
    SZxAppBufferSet_2d1d *buffer_set,
    unsigned char *&encode_pos,
    T *laplacian_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bwidth * size.dim2;
    buffer_set->reset();
    extract_block_mean(cmpkit_set->compressed+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<numBlockRow; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            laplacianProcessBlockRowPrePred(x, size, buffer_set, laplacian_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numBlockRow - 1){
                laplacianProcessBlockRowPrePred(x, size, buffer_set, laplacian_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                laplacianProcessBlockRowPrePred(x, size, buffer_set, laplacian_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZx_laplacian2D_1dMeanbased(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    size_t numblockRow = (size.dim1 - 1) / size.Bwidth + 1;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZxAppBufferSet_2d1d * buffer_set = new SZxAppBufferSet_2d1d(buffer_dim1, buffer_dim2, Buffer_2d, appType::CENTRALDIFF);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, blocks_mean_quant, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            laplacian2DProcessBlocksPrePred(size, numblockRow, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZx_decompress2D_1dMeanbased(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    free(decData);
}

#endif