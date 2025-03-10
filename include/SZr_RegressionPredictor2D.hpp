#ifndef _SZR_REG_PREDICTOR_2D_HPP
#define _SZR_REG_PREDICTOR_2D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZr_app_utils.hpp"
#include "utils.hpp"

template <class T>
void SZr_compress_2dRegression(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_2D * sizeof(float));
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_2D * FLOAT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        const T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate, max_err = 0;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            const T * curr_data_pos = y_data_pos;
            compute_regression_coeffcients_2d(y_data_pos, size_x, size_y, size.dim0_offset, reg_coeff);
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    T pred = predict_regression_2d<T>(i, j, reg_coeff);
                    int err = SZ_quantize(curr_data_pos[j] - pred, errorBound);
                    int abs_err = abs(err);
                    *sign_pos++ = (err < 0);
                    *abs_err_pos++ = abs_err;
                    max_err = max_err > abs_err ? max_err : abs_err;
                }
                curr_data_pos += size.dim0_offset;
            }
            y_data_pos += size.Bsize;
            fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            save_regression_coeff_2d(reg_coeff_pos, reg_coeff);
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
        }
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(reg_coeff);
}

template <class T>
void SZr_decompress_2dRegression(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_2D * sizeof(float));
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_2D * FLOAT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind];
            T * curr_data_pos = y_data_pos;
            extract_regression_coeff(REG_COEFF_SIZE_2D, reg_coeff_pos, reg_coeff);
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
                        T pred = predict_regression_2d<T>(i, j, reg_coeff);
                        curr_data_pos[j] = pred + pred_err_pos[j] * 2 * errorBound;
                    }
                    curr_data_pos += size.dim0_offset;
                    pred_err_pos += size_y;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        curr_data_pos[j] = predict_regression_2d<T>(i, j, reg_coeff);
                    }
                    curr_data_pos += size.dim0_offset;
                }
            }
            y_data_pos += size.Bsize;
            block_ind++;
        }
        x_data_pos += size.Bsize * size.dim2;
    }
    free(signPredError);
    free(signFlag);
    free(reg_coeff);
}

template <class T>
double SZr_mean_2d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZr_decompress_2dRegression(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

double SZr_mean_2d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_2D * sizeof(float));
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    double mean = compute_mean_2d(size, cmpData, signFlag, signPredError, reg_coeff_pos, reg_coeff, errorBound);
    free(signPredError);
    free(signFlag);
    free(reg_coeff);
    return mean;
}

template <class T>
double SZr_mean_2d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            mean = SZr_mean_2d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            printf("Pre-prediction state does not apply to SZr.\n");
            exit(0);
            break;
        }
        case decmpState::postPred:{
            mean = SZr_mean_2d_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

template <class T>
double SZr_variance_2d_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZr_decompress_2dRegression(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZr_variance_2d_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_2D * FLOAT_BYTES) * size.num_blocks;
    unsigned char * reg_coeff_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(REG_COEFF_SIZE_2D * sizeof(float));
    double mean = compute_mean_2d(size, cmpData, signFlag, signPredError, reg_coeff_pos, reg_coeff, errorBound);
    double var = 0;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            extract_regression_coeff(REG_COEFF_SIZE_2D, reg_coeff_pos, reg_coeff);
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
                        T curr_data = predict_regression_2d<T>(i, j, reg_coeff) + (*pred_err_pos++) * 2 * errorBound;
                        var += (curr_data - mean) * (curr_data - mean);
                    }
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        T pred = predict_regression_2d<T>(i, j, reg_coeff);
                        var += (pred - mean) * (pred - mean);
                    }
                }
            }
        }
    }
    free(signPredError);
    free(signFlag);
    free(reg_coeff);
    var /= (size.nbEle - 1);
    return var;
}

template <class T>
double SZr_variance_2d(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZr_variance_2d_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            printf("Pre-prediction state does not apply to SZr.\n");
            exit(0);
            break;
        }
        case decmpState::postPred:{
            var = SZr_variance_2d_postPred<T>(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockRow2PostPred(
    size_t x, DSize_2d& size, SZrCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos,
    size_t buffer_dim0_offset
){
    int block_ind_offset = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    unsigned char * reg_coeff_pos = cmpkit_set->compressed + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + REG_COEFF_SIZE_2D * FLOAT_BYTES * block_ind_offset;
    float * reg_coeff = cmpkit_set->reg_coeff + block_ind_offset * REG_COEFF_SIZE_2D;
    extract_regression_coeff(REG_COEFF_SIZE_2D, reg_coeff_pos, reg_coeff, size.block_dim2);
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int block_ind = block_ind_offset + y;
        int fixed_rate = (int)cmpkit_set->compressed[block_ind];
        if(fixed_rate){
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            for(int i=0; i<size_x; i++){
                memcpy(buffer_start_pos+i*buffer_dim0_offset, cmpkit_set->signPredError+i*size_y, size_y*sizeof(int));
            }
        }
        else{
            for(int i=0; i<size_x; i++){
                memset(buffer_start_pos+i*buffer_dim0_offset, 0, size_y*sizeof(int));
            }
        }
        buffer_start_pos += size.Bsize;
    }
}

template <class T>
inline void dxdyProcessBlockRowPostPred(
    size_t x, DSize_2d size, SZrAppBufferSet_2d *buffer_set,
    SZrCmpBufferSet *cmpkit_set, T *dx_start_pos, T *dy_start_pos,
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
        int * prev_row_pos = isTopRow && i == 0 ? curr_row
                             : i == 0 ? prevBlockBottom_pos
                             : curr_row - buffer_set->buffer_dim0_offset;
        int * next_row_pos = isBottomRow && i == size_x - 1 ? curr_row
                             : i == size_x - 1 ? nextBlockTop_pos
                             : curr_row + buffer_set->buffer_dim0_offset;
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
            float * beta_curr = cmpkit_set->reg_coeff + block_ind * REG_COEFF_SIZE_2D;
            float * beta_left = beta_curr - REG_COEFF_SIZE_2D;
            float * beta_right = beta_curr + REG_COEFF_SIZE_2D;
            float * beta_top = beta_curr - size.block_dim2 * REG_COEFF_SIZE_2D;
            float * beta_bottom = beta_curr + size.block_dim2 * REG_COEFF_SIZE_2D;
            for(int j=0; j<size_y; j++){
                bool isBlockHead = j == 0;
                bool isBlockTail = j == size_y - 1;
                bool isHeadEle = isHeadBlock && isBlockHead;
                bool isTailEle = isTailBlock && isBlockTail;
                int coeff_dy = isHeadEle || isTailEle ? 2 : 1;
                size_t curr_ind = y * size.Bsize + j;
                size_t prev_ind = isHeadEle ? curr_ind : curr_ind - 1;
                size_t next_ind = isTailEle ? curr_ind : curr_ind + 1;
                int dx_integer = next_row_pos[curr_ind] - prev_row_pos[curr_ind];
                int dy_integer = curr_row[next_ind] - curr_row[prev_ind];
                T dx_pred_diff = beta_curr[0] / coeff_dx;
                T dy_pred_diff = beta_curr[1] / coeff_dy;
                if(isBlockTop && !isTopRow){
                    dx_pred_diff = dxdyblockTopPredDiff<T>(j, size.Bsize, beta_curr, beta_top);
                }
                if(isBlockBottom && !isBottomRow){
                    dx_pred_diff = dxdyblockBottomPredDiff<T>(j, size.Bsize, beta_curr, beta_bottom);
                }
                if(isBlockHead && !isHeadBlock){
                    dy_pred_diff = dxdyblockHeadPredDiff<T>(i, size.Bsize, beta_curr, beta_left);
                }
                if(isBlockTail && !isTailBlock){
                    dy_pred_diff = dxdyblockTailPredDiff<T>(i, size.Bsize, beta_curr, beta_right);
                }
                dx_pos[curr_ind] = (dx_integer * errorBound + dx_pred_diff) * coeff_dx;
                dy_pos[curr_ind] = (dy_integer * errorBound + dy_pred_diff) * coeff_dy;
            }
        }
        dx_pos += size.dim2;
        dy_pos += size.dim2;
        curr_row += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdyProcessBlocksPostPred(
    DSize_2d& size,
    SZrCmpBufferSet *cmpkit_set, 
    SZrAppBufferSet_2d *buffer_set,
    unsigned char *encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    int * tempBlockRow = nullptr;
    buffer_set->reset();
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
void SZr_dxdy_2dRegression(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    float * reg_coeff = (float *)malloc(size.num_blocks * REG_COEFF_SIZE_2D * sizeof(float));
    int * Buffer_2d = (int *)malloc(size.Bsize * size.dim2 * 3 * sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    SZrAppBufferSet_2d * buffer_set = new SZrAppBufferSet_2d(size.Bsize, size.dim2, Buffer_2d, appType::CENTRALDIFF);
    SZrCmpBufferSet * cmpkit_set = new SZrCmpBufferSet(cmpData, reg_coeff, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_2D * FLOAT_BYTES) * size.num_blocks;
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
            printf("Pre-prediction state does not apply to SZr.\n");
            exit(0);
            break;
        }
        case decmpState::full:{
            SZr_decompress_2dRegression(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(reg_coeff);
    free(decData);
}

#endif