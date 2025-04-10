#ifndef _SZP_LORENZO_PREDICTOR_2D_1D_HPP
#define _SZP_LORENZO_PREDICTOR_2D_1D_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "typemanager.hpp"
#include "SZp_app_utils.hpp"
#include "utils.hpp"
#include "settings.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZp_compress2D_1dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_2d1d size(dim1, dim2, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim2+1)*sizeof(int));
    quant_buffer[0] = 0;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            int max_err = 0;
            int * block_buffer_pos = buffer_start_pos;
            const T * curr_data_pos = y_data_pos;
            for(int i=0; i<block_size; i++){
                int err = predict_lorenzo_1d(curr_data_pos++, block_buffer_pos++, inver_eb);
                (*sign_pos++) = (err < 0);
                unsigned int abs_err = abs(err);
                (*abs_err_pos++) = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_decompress2D_1dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim2+1)*sizeof(int));
    quant_buffer[0] = 0;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            T * curr_data_pos = y_data_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
            }else{
                for(int i=0; i<block_size; i++){
                    signPredError[i] = 0;
                }
            }
            for(int i=0; i<block_size; i++){
                block_buffer_pos[0] = signPredError[i];
                recover_lorenzo_1d(curr_data_pos++, block_buffer_pos++, errorBound);
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
}

double SZp_mean2D_1dLorenzo_recover2PostPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
            }else{
                for(int j=0; j<block_size; j++) signPredError[j] = 0;
            }
            for(int j=0; j<block_size; j++){
                quant_sum += (size.dim2 - (offset + j)) * signPredError[j];
            }
            offset += block_size;
        }
    }
    free(signPredError);
    free(signFlag);
    double mean = 2 * errorBound * (double)quant_sum / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean2D_1dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress2D_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean2D_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean = -9999;
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean2D_1dLorenzo_recover2PostPred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            // mean = SZp_mean2D_1dLorenzo_recover2PrePred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean2D_1dLorenzo_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZp_variance2D_1dLorenzo_recover2PrePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d1d size(dim1, dim2, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim2+1)*sizeof(int));
    quant_buffer[0] = 0;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * data_pos = quant_buffer + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                for(int j=0; j<block_size; j++){
                    data_pos[0] = signPredError[j];
                    recover_lorenzo_1d(data_pos);
                    int64_t d = static_cast<int64_t>(data_pos[0]);
                    uint64_t d2 = d * d;
                    quant_sum += d;
                    squared_quant_sum += d2;
                    data_pos++;
                }
            }else{
                data_pos[block_size - 1] = data_pos[-1];
                data_pos += block_size;
                int64_t d = static_cast<int64_t>(data_pos[-1]);
                quant_sum += d * block_size;
                uint64_t d2 = d * d;
                squared_quant_sum += static_cast<uint64_t>(block_size) * d2;
            }
        }
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = (2 * errorBound) * (2 * errorBound)* ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
    return var;
}

template <class T>
double SZp_variance2D_1dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress2D_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZp_variance2D_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var = -9999;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZp_variance2D_1dLorenzo_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZp_variance2D_1dLorenzo_recover2PrePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockRow2PostPred(
    size_t x, DSize_2d1d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                for(int j=0; j<block_size; j++){
                    block_buffer_pos[j] = cmpkit_set->signPredError[j];
                }
            }else{
                for(int j=0; j<block_size; j++){
                    block_buffer_pos[j] = 0;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockRow2PrePred(
    size_t x, DSize_2d1d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            }else{
                for(int j=0; j<block_size; j++){
                    cmpkit_set->signPredError[j] = 0;
                }
            }
            for(int j=0; j<block_size; j++){
                block_buffer_pos[0] = cmpkit_set->signPredError[j];
                recover_lorenzo_1d(block_buffer_pos++);
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
    size_t x, DSize_2d1d& size, SZpAppBufferSet_2d1d *buffer_set,
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

// template <class T>
// inline void dxdyProcessBlocksPostPred(
//     DSize_2d1d& size,
//     size_t numBlockRow,
//     SZpCmpBufferSet *cmpkit_set, 
//     SZpAppBufferSet_2d1d *buffer_set,
//     unsigned char *&encode_pos,
//     T *dx_pos, T *dy_pos,
//     double errorBound
// ){
//     size_t BlockRowSize = size.Bwidth * size.dim2;
//     buffer_set->reset();
//     int * tempRow_pos = nullptr;
//     for(size_t x=0; x<numBlockRow; x++){
//         size_t offset = x * BlockRowSize;
//         if(x == 0){
//             recoverBlockRow2PostPred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
//             recoverBlockRow2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
//             dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
//         }else{
//             rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
//             if(x == numBlockRow - 1){
//                 dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
//             }else{
//                 recoverBlockRow2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
//                 dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
//             }
//         }
//     }
// }

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_2d1d& size,
    size_t numBlockRow,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d1d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bwidth * size.dim2;
    buffer_set->reset();
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
void SZp_dxdy_1dLorenzo(
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
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d1d * buffer_set = new SZpAppBufferSet_2d1d(buffer_dim1, buffer_dim2, Buffer_2d, nullptr, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
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
            SZp_decompress2D_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(decData);
}

template <class T>
inline void laplacian2DProcessBlockRowPrePred(
    size_t x, DSize_2d1d& size,
    SZpAppBufferSet_2d1d *buffer_set,
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
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d1d *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bwidth * size.dim2;
    buffer_set->reset();
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<numBlockRow; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            laplacian2DProcessBlockRowPrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numBlockRow - 1){
                laplacian2DProcessBlockRowPrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                laplacian2DProcessBlockRowPrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_laplacian2D_1dLorenzo(
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
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d1d * buffer_set = new SZpAppBufferSet_2d1d(buffer_dim1, buffer_dim2, Buffer_2d, nullptr, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
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
            SZp_decompress2D_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(decData);
}



#endif