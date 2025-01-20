#ifndef _SZP_LORENZO_PREDICTOR_1D_HPP
#define _SZP_LORENZO_PREDICTOR_1D_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "typemanager.hpp"
#include "SZp_app_utils.hpp"
#include "utils.hpp"

template <class T>
void SZp_compress_1dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    DSize_1d size(dim1, dim2, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.dim2+1)*sizeof(int));
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
                int err = predict_lorenzo_1d(curr_data_pos++, block_buffer_pos++, errorBound);
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
void SZp_decompress_1dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_1d size(dim1, dim2, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.dim2+1)*sizeof(int));
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
                memcpy(block_buffer_pos, signPredError, block_size * sizeof(int));
            }else{
                memset(block_buffer_pos, 0, block_size * sizeof(int));
            }
            for(int i=0; i<block_size; i++){
                recover_lorenzo_1d(curr_data_pos++, block_buffer_pos++, errorBound);
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
}

inline void recoverBlockRow2PostPred(
    size_t x, DSize_1d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
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
                for(int i=0; i<block_size; i++){
                    block_buffer_pos[i] = cmpkit_set->signPredError[i];
                }
            }else{
                memset(block_buffer_pos, 0, block_size * sizeof(int));
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

inline void recoverBlockRow2PostPred(
    size_t x, DSize_1d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, int *rowSum, size_t buffer_dim0_offset
){
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        int row_sum = 0;
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
                for(int i=0; i<block_size; i++){
                    block_buffer_pos[i] = cmpkit_set->signPredError[i];
                    row_sum += cmpkit_set->signPredError[i];
                }
            }else{
                memset(block_buffer_pos, 0, block_size * sizeof(int));
            }
            buffer_start_pos += size.Bsize;
        }
        rowSum[x*size.Bwidth+i] = row_sum;
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

inline void recoverBlockRow2PrePred(
    size_t x, DSize_1d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
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
                memcpy(block_buffer_pos, cmpkit_set->signPredError, block_size * sizeof(int));
            }else{
                memset(block_buffer_pos, 0, block_size * sizeof(int));
            }
            for(int i=0; i<block_size; i++){
                recover_lorenzo_1d(block_buffer_pos++);
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

template <class T>
inline void dxdyProcessBlockRowPostPred(
    size_t x, DSize_1d& size, SZpAppBufferSet_1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * buffer_pos = buffer_set->currRow_data_pos;
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    for(int i=0; i<size_x; i++){
        const int * prev_buffer_pos = isTopRow && i == 0 ? buffer_pos
                          : i == 0 ? buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset
                          : buffer_pos - buffer_set->buffer_dim0_offset;
        const int * next_buffer_pos = isBottomRow && i == size_x - 1 ? buffer_pos
                          : i == size_x - 1 ? buffer_set->nextRow_data_pos
                          : buffer_pos + buffer_set->buffer_dim0_offset;
        int coeff_dx = (isTopRow && i == 0) || (isBottomRow && i == size_x - 1) ? 2 : 1;
        int dx_buffer = 0;
        for(size_t j=0; j<size.dim2; j++){
            dx_buffer += (next_buffer_pos[j] - prev_buffer_pos[j]) * coeff_dx;
            dx_pos[j] = dx_buffer * errorBound;
            size_t curr_j = j == 0 ? j + 1 : j;
            size_t next_j = j == size.dim2 - 1 ? j : j + 1;
            dy_pos[j] = (buffer_pos[curr_j] + buffer_pos[next_j]) * errorBound;
        }
        dx_pos += size.dim2;
        dy_pos += size.dim2;
        buffer_pos += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdyProcessBlockRowPrePred(
    size_t x, DSize_1d& size, SZpAppBufferSet_1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * buffer_pos = buffer_set->currRow_data_pos;
    T *dx_pos = dx_start_pos;
    T *dy_pos = dy_start_pos;
    for(int i=0; i<size_x; i++){
        const int * prev_buffer_pos = isTopRow && i == 0 ? buffer_pos
                          : i == 0 ? buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset
                          : buffer_pos - buffer_set->buffer_dim0_offset;
        const int * next_buffer_pos = isBottomRow && i == size_x - 1 ? buffer_pos
                          : i == size_x - 1 ? buffer_set->nextRow_data_pos
                          : buffer_pos + buffer_set->buffer_dim0_offset;
        int coeff_dx = (isTopRow && i == 0) || (isBottomRow && i == size_x - 1) ? 2 : 1;
        for(size_t j=0; j<size.dim2; j++){
            dx_pos[j] = (next_buffer_pos[j] - prev_buffer_pos[j]) * coeff_dx * errorBound;
            size_t prev_j = j == 0 ? j : j - 1;
            size_t next_j = j == size.dim2 - 1 ? j : j + 1;
            int coeff_dy = (j == 0) || (j == size.dim2 - 1) ? 2 : 1;
            dy_pos[j] = (buffer_pos[next_j] - buffer_pos[prev_j]) * coeff_dy * errorBound;
        }
        dx_pos += size.dim2;
        dy_pos += size.dim2;
        buffer_pos += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdyProcessBlocksPostPred(
    DSize_1d& size,
    size_t numBlockRow,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_1d *buffer_set,
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
            recoverBlockRow2PostPred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numBlockRow - 1){
                dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }else{
                recoverBlockRow2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                dxdyProcessBlockRowPostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
}

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_1d& size,
    size_t numBlockRow,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_1d *buffer_set,
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
}

template <class T>
void SZp_dxdy_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_1d size(dim1, dim2, blockSideLength);
    size_t numblockRow = (size.dim1 - 1) / size.Bwidth + 1;
    size_t buffer_dim1 = size.Bwidth;
    size_t buffer_dim2 = size.dim2 + 1;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_1d * buffer_set = new SZpAppBufferSet_1d(buffer_dim1, buffer_dim2, Buffer_2d, nullptr, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdyProcessBlocksPostPred(size, numblockRow, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred(size, numblockRow, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            SZp_decompress_1dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            compute_dxdy(dim1, dim2, decData, dx_pos, dy_pos);
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

inline void heatdisProcessBlockRowPostPred(
    size_t x, DSize_1d& size, Temperature_info& temp_info,
    SZpAppBufferSet_1d *buffer_set, SZpCmpBufferSet *cmpkit_set,
    int next, int iter, bool isTopRow, bool isBottomRow
){
    int bias = (iter & 1) + 1;
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_postpred(x, buffer_set->currRow_data_pos, size, size_x, buffer_set, temp_info, isTopRow, isBottomRow);
    buffer_set->prepare_alternative(size.Bwidth, x, size_x, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset, temp_info.q_W, buffer_set->cmp_buffer);
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        buffer_set->reset_residual();
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            const int * block_buffer_pos = buffer_start_pos;
            int * block_update_pos = update_start_pos;
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int j=0; j<block_size; j++){
                bool flag = y == 0 && j == 1;
                integerize_pred_err(buffer_set, block_buffer_pos++, buffer_set->cmp_buffer+x*size.Bwidth+i, flag, bias, block_update_pos++);
            }
            buffer_start_pos += size.Bsize;
            update_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
        update_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

inline void heatdisCompressBlockRowPostPred(
    size_t x, DSize_1d& size, SZpAppBufferSet_1d *buffer_set,
    SZpCmpBufferSet *cmpkit_set, int next, int iter
){
    int bias = (iter & 1) + 1;
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = encode_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        buffer_set->reset_residual();
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int * block_update_pos = update_start_pos;
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int j=0; j<block_size; j++){
                int err = *block_update_pos++;
                *sign_pos++ = (err < 0);
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
            update_start_pos += size.Bsize;
        }
        update_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
    size_t increment = encode_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void heatdisProcessBlockRowPrePred(
    size_t x, DSize_1d& size, Temperature_info& temp_info,
    SZpAppBufferSet_1d *buffer_set, SZpCmpBufferSet *cmpkit_set,
    int next, int iter, bool isTopRow, bool isBottomRow
){
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_prepred(buffer_set->currRow_data_pos, size, buffer_set->buffer_dim0_offset, temp_info, isTopRow, isBottomRow);
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            const int * block_buffer_pos = buffer_start_pos;
            int * block_update_pos = update_start_pos;
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int j=0; j<block_size; j++){
                integerize_quant(buffer_set, block_buffer_pos++, block_update_pos++);
            }
            buffer_start_pos += size.Bsize;
            update_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
        update_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

inline void heatdisCompressBlockRowPrePred(
    size_t x, DSize_1d& size, SZpAppBufferSet_1d *buffer_set,
    SZpCmpBufferSet *cmpkit_set, int next, int iter
){
    int block_ind = x * size.Bwidth * size.block_dim2;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = encode_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int * block_update_pos = update_start_pos;
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int j=0; j<block_size; j++){
                int err = predict_lorenzo_1d(block_update_pos++);
                *sign_pos++ = (err < 0);
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
            update_start_pos += size.Bsize;
        }
        update_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
    size_t increment = encode_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void heatdisUpdatePostPred(
    DSize_1d& size,
    size_t numblockRow,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_1d *buffer_set,
    Temperature_info& temp_info,
    int current, int next,
    int iter
){
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    for(size_t x=0; x<numblockRow; x++){
        if(x == 0){
            recoverBlockRow2PostPred(x, size, cmpData, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->rowSum, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->rowSum, buffer_set->buffer_dim0_offset);
            heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, true, false);
            heatdisCompressBlockRowPostPred(x, size, buffer_set, cmpkit_set, next, iter);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numblockRow - 1){
                heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, false, true);
                heatdisCompressBlockRowPostPred(x, size, buffer_set, cmpkit_set, next, iter);
            }else{
                recoverBlockRow2PostPred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->rowSum, buffer_set->buffer_dim0_offset);
                heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, false, false);
                heatdisCompressBlockRowPostPred(x, size, buffer_set, cmpkit_set, next, iter);
            }
        }
    }
}

inline void heatdisUpdatePrePred(
    DSize_1d& size,
    size_t numblockRow,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_1d *buffer_set,
    Temperature_info& temp_info,
    int current, int next,
    int iter
){
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    for(size_t x=0; x<numblockRow; x++){
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpData, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, true, false);
            heatdisCompressBlockRowPrePred(x, size, buffer_set, cmpkit_set, next, iter);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numblockRow - 1){
                heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, false, true);
                heatdisCompressBlockRowPrePred(x, size, buffer_set, cmpkit_set, next, iter);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, cmpkit_set, next, iter, false, false);
                heatdisCompressBlockRowPrePred(x, size, buffer_set, cmpkit_set, next, iter);
            }
        }
    }
}

template <class T>
inline void heatdisUpdatePostPred(
    DSize_1d& size,
    size_t numblockRow,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_1d *buffer_set,
    Temperature_info& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    for(int iter=1; iter<=max_iter; iter++){
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePostPred(size, numblockRow, cmpkit_set, buffer_set, temp_info, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter % ht_plot_gap == 0){
                SZp_decompress_1dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
                std::string h_name = "/Users/xuanwu/github/backup/homoApplication/plot/ht_data/h1d.post." + std::to_string(iter);
                writefile(h_name.c_str(), h, size.nbEle);
                size_t cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
                printf("postpred iter %d: cr = %.2f\n", iter, 1.0 * size.nbEle * sizeof(T) / cmpSize);
            }
        }
    }
    printf("postpred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdatePrePred(
    DSize_1d& size,
    size_t numblockRow,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_1d *buffer_set,
    Temperature_info& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    for(int iter=1; iter<=max_iter; iter++){
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePrePred(size, numblockRow, cmpkit_set, buffer_set, temp_info, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter % ht_plot_gap == 0){
                SZp_decompress_1dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
                std::string h_name = "/Users/xuanwu/github/backup/homoApplication/plot/ht_data/h1d.pre." + std::to_string(iter);
                writefile(h_name.c_str(), h, size.nbEle);
                size_t cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
                printf("prepred iter %d: cr = %.2f\n", iter, 1.0 * size.nbEle * sizeof(T) / cmpSize);
            }
        }
    }
    printf("prepred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdateDOC(
    DSize_1d& size,
    size_t dim1_padded,
    size_t dim2_padded,
    Temperature_info& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t nbEle_padded = dim1_padded * dim2_padded;
    T * h = (T *)malloc(nbEle_padded * sizeof(T));
    T * h2 = (T *)malloc(nbEle_padded * sizeof(T));
    unsigned char * compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    HeatDis heatdis(temp_info.src_temp, temp_info.wall_temp, temp_info.ratio, size.dim1, size.dim2);
    heatdis.initData(h, h2, temp_info.init_temp);
    size_t cmpSize = 0;
    SZp_compress_1dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
    T * tmp = nullptr;
    for(int iter=1; iter<=max_iter; iter++){
        clock_gettime(CLOCK_REALTIME, &start);
        SZp_decompress_1dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound);
        heatdis.reset_source(h, h2);
        heatdis.iterate(h, h2, tmp);
        SZp_compress_1dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter % ht_plot_gap == 0){
                std::string h_name = "/Users/xuanwu/github/backup/homoApplication/plot/ht_data/h1d.doc." + std::to_string(iter);
                writefile(h_name.c_str(), h, nbEle_padded);
                printf("doc iter %d: cr = %.2f\n", iter, 1.0 * nbEle_padded * sizeof(T) / cmpSize);
            }
        }
    }
    printf("doc elapsed_time = %.6f\n", elapsed_time);
    free(h);
    free(h2);
    free(compressed);
}

template <class T>
void SZp_heatdis_1dLorenzo(
    unsigned char *cmpDataBuffer, size_t dim1, size_t dim2,
    int blockSideLength, int max_iter, size_t& cmpSize,
    float source_temp, float wall_temp, float init_temp, double ratio,
    double errorBound, decmpState state, bool verb
){
    DSize_1d size(dim1, dim2, blockSideLength);
    size_t numblockRow = (size.dim1 - 1) / size.Bwidth + 1;
    size_t dim1_padded = size.dim1 + 2;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    size_t nbEle_padded = dim1_padded * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    int * Buffer_1d = (int *)malloc(buffer_dim2 * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * signPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));

    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        cmpData[i] = (unsigned char *)malloc(size.nbEle * sizeof(T)*2);
        offsets[i] = (int *)malloc(numblockRow * sizeof(int));
    }
    memcpy(cmpData[0], cmpDataBuffer, size.nbEle * sizeof(T));

    size_t prefix_length = 0;
    int block_index = 0;
    for(size_t x=0; x<numblockRow; x++){
        offsets[0][x] = prefix_length;
        offsets[1][x] = 0;
        int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
        for(int i=0; i<size_x; i++){
            for(size_t y=0; y<size.block_dim2; y++){
                int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int cmp_block_sign_length = (block_size + 7) / 8;
                int fixed_rate = (int)cmpDataBuffer[block_index++];
                size_t savedbitsbytelength = compute_encoding_byteLength(block_size, fixed_rate);
                if(fixed_rate)
                    prefix_length += (cmp_block_sign_length + savedbitsbytelength);
            }
        }
    }
    Temperature_info temp_info(source_temp, wall_temp, init_temp, ratio, errorBound);
    SZpAppBufferSet_1d * buffer_set = new SZpAppBufferSet_1d(buffer_dim1, buffer_dim2, Buffer_2d, Buffer_1d, appType::HEATDIS);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, offsets, absPredError, signPredError, signFlag);

    switch(state){
        case decmpState::postPred:{
            temp_info.prepare_src_row(size.dim2, buffer_set->decmp_buffer, buffer_set->lorenzo_buffer);
            heatdisUpdatePostPred<T>(size, numblockRow, cmpkit_set, buffer_set, temp_info, errorBound, max_iter, verb);
            break;
        }
        case decmpState::prePred:{
            heatdisUpdatePrePred<T>(size, numblockRow, cmpkit_set, buffer_set, temp_info, errorBound, max_iter, verb);
            break;
        }
        case decmpState::full:{
            heatdisUpdateDOC<T>(size, dim1_padded, buffer_dim2, temp_info, errorBound, max_iter, verb);
            break;
        }
    }

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_2d);
    free(Buffer_1d);
    free(absPredError);
    free(signPredError);
    free(signFlag);
    for(int i=0; i<2; i++){
        free(cmpData[i]);
        free(offsets[i]);
    }
    free(cmpData);
    free(offsets);
}

#endif