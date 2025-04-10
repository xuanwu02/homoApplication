#ifndef _SZP_LORENZO_PREDICTOR_2D_HPP
#define _SZP_LORENZO_PREDICTOR_2D_HPP

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
void SZp_compress_2dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim0_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            int max_err = 0;
            int * block_buffer_pos = buffer_start_pos;
            const T * curr_data_pos = y_data_pos;
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    int err = predict_lorenzo_2d(curr_data_pos++, curr_buffer_pos++, buffer_dim0_offset, inver_eb);
                    (*sign_pos++) = (err < 0);
                    unsigned int abs_err = abs(err);
                    (*abs_err_pos++) = abs_err;
                    max_err = max_err > abs_err ? max_err : abs_err;
                }
                block_buffer_pos += buffer_dim0_offset;
                curr_data_pos += size.dim0_offset - size_y;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                cmpData_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, cmpData_pos, fixed_rate);
                cmpData_pos += savedbitsbyteLength;
            }
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    cmpSize = cmpData_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_decompress_2dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim0_offset = size.dim2 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + buffer_dim0_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            T * curr_data_pos = y_data_pos;
            if(!fixed_rate){
                memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
            }
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[0] = signPredError[i*size_y+j];
                    recover_lorenzo_2d(curr_data_pos++, curr_buffer_pos++, buffer_dim0_offset, errorBound);
                }
                block_buffer_pos += buffer_dim0_offset;
                curr_data_pos += size.dim0_offset - size_y;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
        x_data_pos += size.Bsize * size.dim2;
    }
    free(pred_buffer);
    free(signPredError);
    free(signFlag);
}

double SZp_mean_2dLorenzo_recover2PostPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int index_x = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int index_y = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                const int * pred_pos = signPredError;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (*pred_pos++);
                    }
                }
            }
            index_y += size.Bsize;
        }
        index_x += size.Bsize;
    }
    free(signPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

double SZp_mean_2dLorenzo_recover2PrePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim0_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(!fixed_rate){
                memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
            }
            int * curr_buffer_pos = block_buffer_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[0] = signPredError[i*size_y+j];
                    recover_lorenzo_2d(quant_sum, curr_buffer_pos++, buffer_dim0_offset);
                }
                curr_buffer_pos += buffer_dim0_offset - size_y;
            }
            buffer_start_pos += size.Bsize;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean_2dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean_2dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean_2dLorenzo_recover2PostPred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            mean = SZp_mean_2dLorenzo_recover2PrePred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean_2dLorenzo_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZp_variance_2dLorenzo_postPredMean(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim0_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int index_x = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int index_y = 0;
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                const int * pred_pos = signPredError;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (*pred_pos++);
                    }
                }
            }else{
                memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
            }
            int * curr_buffer_pos = block_buffer_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[0] = signPredError[i*size_y+j];
                    int curr_quant = recover_lorenzo_2d_verb(curr_buffer_pos++, buffer_dim0_offset);
                    int64_t d = static_cast<int64_t>(curr_quant);
                    uint64_t d2 = d * d;
                    squared_quant_sum += d2;
                }
                curr_buffer_pos += buffer_dim0_offset - size_y;
            }
            buffer_start_pos += size.Bsize;
            index_y += size.Bsize;
        }
        index_x += size.Bsize;
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = (2 * errorBound) * (2 * errorBound) * ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
    return var;
}

double SZp_variance_2dLorenzo_prePredMean(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim0_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
            }else{
                memset(signPredError, 0, size.max_num_block_elements*sizeof(int));                
            }
            int * curr_buffer_pos = block_buffer_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[0] = signPredError[i*size_y+j];
                    int curr_quant = recover_lorenzo_2d_verb(curr_buffer_pos++, buffer_dim0_offset);
                    int64_t d = static_cast<int64_t>(curr_quant);
                    uint64_t d2 = d * d;
                    quant_sum += d;
                    squared_quant_sum += d2;
                }
                curr_buffer_pos += buffer_dim0_offset - size_y;
            }
            buffer_start_pos += size.Bsize;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = (2 * errorBound) * (2 * errorBound) * ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
    return var;
}

template <class T>
double SZp_variance_2dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZp_variance_2dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZp_variance_2dLorenzo_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZp_variance_2dLorenzo_prePredMean(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            var = SZp_variance_2dLorenzo_postPredMean(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockRow2PostPred_IntBuffer(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind++];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                // memset(curr_buffer_pos, 0, size_y*sizeof(int));
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = 0;
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
                // memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = data_pos[j];
                }
                curr_buffer_pos += buffer_dim0_offset;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void recoverBlockRow2PostPred_FltBuffer(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    T *buffer_data_pos, double errorBound, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    T * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        T * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind++];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                memset(curr_buffer_pos, 0, size_y*sizeof(int));
                // for(int j=0; j<size_y; j++){
                //     curr_buffer_pos[j] = 0;
                // }
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
                // memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = data_pos[j] * errorBound * 2;
                }
                curr_buffer_pos += buffer_dim0_offset;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockRow2PrePred_IntBuffer(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset, int *decmp_buffer
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int fixed_rate = (int)cmpData[block_ind++];
        if(fixed_rate){
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
        }
        else{
            // memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
            for(int i=0; i<block_size; i++) cmpkit_set->signPredError[i] = 0;
        }
        // // legacy
        // int * curr_buffer_pos = buffer_start_pos;
        // int * quant_pos = buffer_start_pos;
        // int * data_pos = cmpkit_set->signPredError;
        // for(int i=0; i<size_x; i++){
        //     memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
        //     curr_buffer_pos += buffer_dim0_offset;
        //     data_pos += size_y;
        //     for(int j=0; j<size_y; j++){
        //         recover_lorenzo_2d(quant_pos+j, buffer_dim0_offset);
        //     }
        //     quant_pos += buffer_dim0_offset;
        // }
        int * curr_buffer_pos = buffer_start_pos;
        for(int i=0; i<size_x; i++){
            for(int j=0; j<size_y; j++){
                curr_buffer_pos[0] = cmpkit_set->signPredError[i*size_y+j];
                recover_lorenzo_2d(curr_buffer_pos++, buffer_dim0_offset);
            }
            curr_buffer_pos += buffer_dim0_offset - size_y;
        }
        buffer_start_pos += size.Bsize;
    }
    memcpy(decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_dim0_offset-1, buffer_dim0_offset*sizeof(int));
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void recoverBlockRow2PrePred_FltBuffer(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    T *buffer_data_pos, T *decmp_buffer,
    double errorBound, size_t buffer_dim0_offset
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    T * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int fixed_rate = (int)cmpData[block_ind++];
        if(fixed_rate){
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
        }
        else{
            // memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
            for(int i=0; i<block_size; i++) cmpkit_set->signPredError[i] = 0;
        }
        // // legacy
        // T * curr_buffer_pos = buffer_start_pos;
        // T * quant_pos = buffer_start_pos;
        // T * data_pos = cmpkit_set->signPredError;
        // for(int i=0; i<size_x; i++){
        //     memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
        //     curr_buffer_pos += buffer_dim0_offset;
        //     data_pos += size_y;
        //     for(int j=0; j<size_y; j++){
        //         recover_lorenzo_2d(quant_pos+j, buffer_dim0_offset);
        //     }
        //     quant_pos += buffer_dim0_offset;
        // }
        T * curr_buffer_pos = buffer_start_pos;
        for(int i=0; i<size_x; i++){
            for(int j=0; j<size_y; j++){
                curr_buffer_pos[0] = cmpkit_set->signPredError[i*size_y+j] * errorBound * 2;
                recover_lorenzo_2d(curr_buffer_pos++, buffer_dim0_offset);
            }
            curr_buffer_pos += buffer_dim0_offset - size_y;
        }
        buffer_start_pos += size.Bsize;
    }
    memcpy(decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_dim0_offset-1, buffer_dim0_offset*sizeof(T));
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockRowPostPred_IntBuffer(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<int> *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const int * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        int dx_buffer = 0;
        const int * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const int * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            dx_buffer += curr_row[j] + next_row[j];
            *dx_pos++ = dx_buffer * errorBound;
            buffer_set->dy_buffer[j] += curr_row[j] + curr_row[j+1];
            *dy_pos++ = buffer_set->dy_buffer[j] * errorBound;
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockRowPostPred_FltBuffer(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<T> *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const T * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const T * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const T * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        T dx_buffer = 0;
        const T * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const T * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            dx_buffer += curr_row[j] + next_row[j];
            *dx_pos++ = dx_buffer;
            buffer_set->dy_buffer[j] += curr_row[j] + curr_row[j+1];
            *dy_pos++ = buffer_set->dy_buffer[j];
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockRowPrePred_IntBuffer(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<int> *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
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
inline void dxdyProcessBlockRowPrePred_FltBuffer(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<T> *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const T * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const T * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const T * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        const T * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const T * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            *dx_pos++ = next_row[j] - prev_row[j];
            *dy_pos++ = curr_row[j+1] - curr_row[j-1];
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlocksPostPred_IntBuffer(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<int> *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PostPred_IntBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else if(x == size.block_dim1 - 1){
            buffer_set->currRow_data_pos = buffer_set->nextRow_data_pos;
            dxdyProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
        }else{
            std::swap(buffer_set->currRow_data_pos, buffer_set->nextRow_data_pos);
            recoverBlockRow2PostPred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdyProcessBlocksPostPred_FltBuffer(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<T> *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PostPred_FltBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else if(x == size.block_dim1 - 1){
            buffer_set->currRow_data_pos = buffer_set->nextRow_data_pos;
            dxdyProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
        }else{
            std::swap(buffer_set->currRow_data_pos, buffer_set->nextRow_data_pos);
            recoverBlockRow2PostPred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdyProcessBlocksPrePred_IntBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<int> *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred_IntBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(int));
            recoverBlockRow2PrePred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            dxdyProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(int));
                recoverBlockRow2PrePred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
                dxdyProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdyProcessBlocksPrePred_FltBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<T> *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    T * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred_FltBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
            memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(T));
            recoverBlockRow2PrePred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos,  buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
            dxdyProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(T));
                recoverBlockRow2PrePred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
                dxdyProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_dxdy_2dLorenzo_IntBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_1d = (int *)malloc(buffer_dim2 * sizeof(int));
    int * Buffer_2d = (int *)malloc((buffer_size * 3 + buffer_dim2 * 2) * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d<int> * buffer_set = new SZpAppBufferSet_2d<int>(buffer_dim1, buffer_dim2, Buffer_2d, Buffer_1d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdyProcessBlocksPostPred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("doc_recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_dxdy(dim1, dim2, decData, dx_pos, dy_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("doc_process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_2d);
    free(Buffer_1d);
    free(signPredError);
    free(signFlag);
    free(decData);
}

template <class T>
void SZp_dxdy_2dLorenzo_FltBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_1d = (int *)malloc(buffer_dim2 * sizeof(int));
    T * Buffer_2d = (T *)malloc((buffer_size * 3 + buffer_dim2 * 2) * sizeof(T));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d<T> * buffer_set = new SZpAppBufferSet_2d<T>(buffer_dim1, buffer_dim2, Buffer_2d, Buffer_1d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            dxdyProcessBlocksPostPred_FltBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            dxdyProcessBlocksPrePred_FltBuffer(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("doc_recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_dxdy(dim1, dim2, decData, dx_pos, dy_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("doc_process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_2d);
    free(Buffer_1d);
    free(signPredError);
    free(signFlag);
    free(decData);
}

// 03/06/2025
template <class T>
inline void laplacianProcessBlockRowPrePred_IntBuffer(
    size_t x, DSize_2d& size,
    SZpAppBufferSet_2d<int> *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
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
inline void laplacianProcessBlockRowPostPred_IntBuffer(
    size_t x, DSize_2d& size,
    SZpAppBufferSet_2d<int> *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * laplacian_pos = result_start_pos;
    const int * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        int x_buffer = 0, x_buffer_2 = 0;
        buffer_set->dy_buffer[0] += curr_row[0];
        const int * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            buffer_set->dy_buffer[j+1] += curr_row[j+1];
            x_buffer += curr_row[j];
            x_buffer_2 += next_row[j];
            *laplacian_pos++ = (buffer_set->dy_buffer[j+1] - buffer_set->dy_buffer[j] + x_buffer_2 - x_buffer) * errorBound * 2;
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
    // int * buffer_start_pos = buffer_set->currRow_data_pos;
    // for(size_t y=0; y<size.block_dim2; y++){
    //     int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
    //     int * block_buffer_pos = buffer_start_pos;
    //     T * block_laplacian_pos = laplacian_pos;
    //     for(int i=0; i<size_x; i++){
    //         int * curr_buffer_pos = block_buffer_pos;
    //         T * curr_laplacian_pos = block_laplacian_pos;
    //         for(int j=0; j<size_y; j++){
    //             *curr_laplacian_pos++ = (curr_buffer_pos[-1] + 
    //                                 curr_buffer_pos[1] + 
    //                                 curr_buffer_pos[-buffer_set->buffer_dim0_offset] +
    //                                 curr_buffer_pos[buffer_set->buffer_dim0_offset] -
    //                                 4 * curr_buffer_pos[0]) * errorBound * 2;
    //             curr_buffer_pos++;
    //         }
    //         block_buffer_pos += buffer_set->buffer_dim0_offset;
    //         block_laplacian_pos += size.dim2;
    //     }
    //     buffer_start_pos += size.Bsize;
    //     laplacian_pos += size.Bsize;
    // }
}

template <class T>
inline void laplacianProcessBlockRowPrePred_FltBuffer(
    size_t x, DSize_2d& size,
    SZpAppBufferSet_2d<T> *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const T * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const T * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(T));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(T));
    T * laplacian_pos = result_start_pos;
    const T * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        const T * prev_row = curr_row - buffer_set->buffer_dim0_offset;
        const T * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            *laplacian_pos++ = curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j];
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void laplacianProcessBlockRowPostPred_FltBuffer(
    size_t x, DSize_2d& size,
    SZpAppBufferSet_2d<T> *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const T * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const T * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    T * laplacian_pos = result_start_pos;
    const T * curr_row = buffer_set->currRow_data_pos;
    for(int i=0; i<size_x; i++){
        T x_buffer = 0, x_buffer_2 = 0;
        buffer_set->dy_buffer[0] += curr_row[0];
        const T * next_row = curr_row + buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            buffer_set->dy_buffer[j+1] += curr_row[j+1];
            x_buffer += curr_row[j];
            x_buffer_2 += next_row[j];
            *laplacian_pos++ = buffer_set->dy_buffer[j+1] - buffer_set->dy_buffer[j] + x_buffer_2 - x_buffer;
        }
        curr_row += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void laplacianProcessBlocksPrePred_IntBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<int> *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred_IntBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(int));
            recoverBlockRow2PrePred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            laplacianProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(int));
                recoverBlockRow2PrePred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
                laplacianProcessBlockRowPrePred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPostPred_IntBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<int> *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PostPred_IntBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
            laplacianProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                recoverBlockRow2PostPred_IntBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset);
                laplacianProcessBlockRowPostPred_IntBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPrePred_FltBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<T> *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    T * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PrePred_FltBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
            memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(T));
            recoverBlockRow2PrePred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos,  buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
            laplacianProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextRow_data_pos - buffer_set->buffer_dim0_offset - 1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset * sizeof(T));
                recoverBlockRow2PrePred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->decmp_buffer, errorBound, buffer_set->buffer_dim0_offset);
                laplacianProcessBlockRowPrePred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPostPred_FltBuffer(
    DSize_2d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_2d<T> *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockRowSize = size.Bsize * size.dim2;
    buffer_set->reset();
    T * tempRow_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockRowSize;
        if(x == 0){
            recoverBlockRow2PostPred_FltBuffer(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PostPred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
            laplacianProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                recoverBlockRow2PostPred_FltBuffer(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, errorBound, buffer_set->buffer_dim0_offset);
                laplacianProcessBlockRowPostPred_FltBuffer(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}
template <class T>
void SZp_laplacian_2dLorenzo_IntBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc((buffer_size * 3 + buffer_dim2 * 2) * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d<int> * buffer_set = new SZpAppBufferSet_2d<int>(buffer_dim1, buffer_dim2, Buffer_2d, nullptr, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            laplacianProcessBlocksPostPred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred_IntBuffer(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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

template <class T>
void SZp_laplacian_2dLorenzo_FltBuffer(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_1d = (int *)malloc(buffer_dim2 * sizeof(int));
    T * Buffer_2d = (T *)malloc((buffer_size * 3 + buffer_dim2 * 2) * sizeof(T));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_2d<T> * buffer_set = new SZpAppBufferSet_2d<T>(buffer_dim1, buffer_dim2, Buffer_2d, Buffer_1d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            laplacianProcessBlocksPostPred_FltBuffer(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred_FltBuffer(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(Buffer_1d);
    free(signPredError);
    free(signFlag);
    free(decData);
}

inline void heatdisRecoverBlockRow2PostPred(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset,
    int *decmp_buffer, int *rowSum, int *colSum
){
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind++];
        size_t row_ind, col_ind;
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                memset(curr_buffer_pos, 0, size_y*sizeof(int));
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
                memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
                row_ind = x * size.Bsize + i;
                for(int j=0; j<size_y; j++){
                    col_ind = y * size.Bsize + j;
                    colSum[col_ind] += curr_buffer_pos[j];
                    rowSum[row_ind] += curr_buffer_pos[j];
                }
                curr_buffer_pos += buffer_dim0_offset;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
}

inline void heatdisRecoverBlockRow2PrePred(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    SZpCmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t buffer_dim0_offset,
    int *decmp_buffer
){
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int fixed_rate = (int)cmpData[block_ind++];
        if(fixed_rate){
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
        }
        else{
            // memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
            for(int i=0; i<block_size; i++) cmpkit_set->signPredError[i] = 0;
        }
        int * curr_buffer_pos = buffer_start_pos;
        for(int i=0; i<size_x; i++){
            for(int j=0; j<size_y; j++){
                curr_buffer_pos[0] = cmpkit_set->signPredError[i*size_y+j];
                recover_lorenzo_2d(curr_buffer_pos++, buffer_dim0_offset);
            }
            curr_buffer_pos += buffer_dim0_offset - size_y;
        }
        buffer_start_pos += size.Bsize;
    }
    memcpy(decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_dim0_offset-1, buffer_dim0_offset*sizeof(int));
}

inline void heatdisProcessBlockRowPostPred(
    size_t x, DSize_2d& size, TempInfo2D& temp_info,
    SZpAppBufferSet_2d<int> *buffer_set, int iter, int bias,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_postpred(x, buffer_set->currRow_data_pos, size, size_x, buffer_set, temp_info, isTopRow, isBottomRow);
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        bool flag = (isTopRow && i == 1);
        for(size_t j=0; j<size.dim2; j++){
            integerize_pred_err(buffer_set, buffer_start_pos++, buffer_set->cmp_buffer+j, flag, bias, update_start_pos++);
        }
        buffer_start_pos += 2;
        update_start_pos += 2;
    }
}

inline void heatdisProcessBlockRowPrePred(
    size_t x, DSize_2d& size, TempInfo2D& temp_info,
    SZpAppBufferSet_2d<int> *buffer_set, int iter, int bias,
    bool isTopRow, bool isBottomRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_prepred(buffer_set->currRow_data_pos, size, size_x, buffer_set->buffer_dim0_offset, temp_info, isTopRow, isBottomRow);
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    int * update_start_pos = buffer_set->updateRow_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        const int * block_buffer_pos = buffer_start_pos;
        int * block_update_pos = update_start_pos;
        for(int i=0; i<size_x; i++){
            const int * curr_buffer_pos = block_buffer_pos;
            int * curr_update_pos = block_update_pos;
            for(int j=0; j<size_y; j++){
                integerize_quant(curr_buffer_pos++, curr_update_pos++, buffer_set->buffer_dim0_offset, bias);
            }
            block_buffer_pos += buffer_set->buffer_dim0_offset;
            block_update_pos += buffer_set->buffer_dim0_offset;
        }
        buffer_start_pos += size_y;
        update_start_pos += size_y;
    }
}

inline void compressBlockRowFromPostPred(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<int> *buffer_set,
    SZpCmpBufferSet *cmpkit_set, int current, int next,
    int iter, bool isTopRow
){
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int block_ind = x * size.block_dim2;
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = cmpData_pos;
    const int * update_start_pos = buffer_set->updateRow_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        const int * block_err_pos = update_start_pos;
        unsigned char * sign_pos = cmpkit_set->signFlag;
        unsigned int * abs_err_pos = cmpkit_set->absPredError;
        int abs_err, max_err = 0;
        for(int i=0; i<size_x; i++){
            const int * curr_err_pos = block_err_pos;
            for(int j=0; j<size_y; j++){
                int err = *curr_err_pos++;
                *sign_pos++ = (err < 0);
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            block_err_pos += buffer_set->buffer_dim0_offset;
        }
        update_start_pos += size_y;
        int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
        cmpData[block_ind++] = (unsigned char)fixed_rate;
        if(fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, cmpData_pos);
            cmpData_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, cmpData_pos, fixed_rate);
            cmpData_pos += savedbitsbyteLength;
        }
    }
    size_t increment = cmpData_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void compressBlockRowFromPrePred(
    size_t x, DSize_2d& size, SZpAppBufferSet_2d<int> *buffer_set,
    SZpCmpBufferSet *cmpkit_set, int current, int next,
    int iter, bool isTopRow
){
    buffer_set->set_cmp_buffer(isTopRow);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    int block_ind = x * size.block_dim2;
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = cmpData_pos;
    const int * update_start_pos = buffer_set->updateRow_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        const int * block_quant_pos = update_start_pos;
        unsigned char * sign_pos = cmpkit_set->signFlag;
        unsigned int * abs_err_pos = cmpkit_set->absPredError;
        int abs_err, max_err = 0;
        for(int i=0; i<size_x; i++){
            const int * curr_quant_pos = block_quant_pos;
            for(int j=0; j<size_y; j++){
                int err = predict_lorenzo_2d(curr_quant_pos++, buffer_set->buffer_dim0_offset);
                *sign_pos++ = (err < 0);
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            block_quant_pos += buffer_set->buffer_dim0_offset;
        }
        update_start_pos += size_y;
        int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
        cmpData[block_ind++] = (unsigned char)fixed_rate;
        if(fixed_rate){
            unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, cmpData_pos);
            cmpData_pos += signbyteLength;
            unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, cmpData_pos, fixed_rate);
            cmpData_pos += savedbitsbyteLength;
        }
    }
    buffer_set->save_cmp_buffer_buttom(size_x);
    size_t increment = cmpData_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void heatdisUpdatePostPred(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_2d<int> *buffer_set,
    TempInfo2D& temp_info,
    int current, int next,
    int iter
){
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int bias = ((iter - 1) & 1) + 1;
    // int bias = 0;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        if(x == 0){
            heatdisRecoverBlockRow2PostPred(x, size, cmpData, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset, nullptr, buffer_set->rowSum, buffer_set->colSum);
            heatdisRecoverBlockRow2PostPred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, nullptr, buffer_set->rowSum, buffer_set->colSum);
            heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, iter, bias, true, false);
            compressBlockRowFromPostPred(x, size, buffer_set, cmpkit_set, current, next, iter, true);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, iter, bias, false, true);
                compressBlockRowFromPostPred(x, size, buffer_set, cmpkit_set, current, next, iter, false);
            }else{
                heatdisRecoverBlockRow2PostPred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, nullptr, buffer_set->rowSum, buffer_set->colSum);
                heatdisProcessBlockRowPostPred(x, size, temp_info, buffer_set, iter, bias, false, false);
                compressBlockRowFromPostPred(x, size, buffer_set, cmpkit_set, current, next, iter, false);
            }
        }
    }
}

inline void heatdisUpdatePrePred(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_2d<int> *buffer_set,
    TempInfo2D& temp_info,
    int current, int next,
    int iter
){
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int bias = ((iter - 1) & 1) + 1;
    // int bias = 0;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        if(x == 0){
            heatdisRecoverBlockRow2PrePred(x, size, cmpData, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            memcpy(buffer_set->nextRow_data_pos-buffer_set->buffer_dim0_offset-1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset*sizeof(int));
            heatdisRecoverBlockRow2PrePred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
            heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, iter, bias, true, false);
            compressBlockRowFromPrePred(x, size, buffer_set, cmpkit_set, current, next, iter, true);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == size.block_dim1 - 1){
                heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, iter, bias, false, true);
                compressBlockRowFromPrePred(x, size, buffer_set, cmpkit_set, current, next, iter, false);
            }else{
                memcpy(buffer_set->nextRow_data_pos-buffer_set->buffer_dim0_offset-1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset*sizeof(int));
                heatdisRecoverBlockRow2PrePred(x+1, size, cmpData, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, buffer_set->buffer_dim0_offset, buffer_set->decmp_buffer);
                heatdisProcessBlockRowPrePred(x, size, temp_info, buffer_set, iter, bias, false, false);
                compressBlockRowFromPrePred(x, size, buffer_set, cmpkit_set, current, next, iter, false);
            }
        }
    }
}

// this is problematic for large sizes
template <class T>
inline void heatdisUpdatePostPred(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_2d<int> *buffer_set,
    TempInfo2D& temp_info,
    double errorBound,
    int max_iter,
    bool record
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
            if(record){
                SZp_decompress_2dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
                std::string h_name = heatdis2d_data_dir + "/h.L2.post." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, size.nbEle);
            }
            cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
            printf("postpred iter %d: cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / cmpSize);
            printf("postpred iter %d: time = %.6f\n", iter-1, elapsed_time);
            fflush(stdout);
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePostPred(size, cmpkit_set, buffer_set, temp_info, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZp_decompress_2dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
        std::string h_name = heatdis2d_data_dir + "/h.L2.post." + std::to_string(iter);
        writefile(h_name.c_str(), h, size.nbEle);
        cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
        printf("postpred exit cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / cmpSize);
    }
    printf("postpred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdatePrePred(
    DSize_2d& size,
    SZpCmpBufferSet *cmpkit_set,
    SZpAppBufferSet_2d<int> *buffer_set,
    TempInfo2D& temp_info,
    double errorBound,
    int max_iter,
    bool record
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
            if(record){
                SZp_decompress_2dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
                std::string h_name = heatdis2d_data_dir + "/h.L2.pre." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, size.nbEle);
            }
            cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
            printf("prepred iter %d: cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / cmpSize);
            printf("prepred iter %d: time = %.6f\n", iter-1, elapsed_time);
            fflush(stdout);
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePrePred(size, cmpkit_set, buffer_set, temp_info, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZp_decompress_2dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
        std::string h_name = heatdis2d_data_dir + "/h.L2.pre." + std::to_string(iter);
        writefile(h_name.c_str(), h, size.nbEle);
        cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
        printf("prepred exit cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / cmpSize);
    }
    printf("prepred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdateDOC(
    DSize_2d& size,
    size_t dim1_padded,
    size_t dim2_padded,
    TempInfo2D& temp_info,
    double errorBound,
    int max_iter,
    bool record
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    size_t nbEle_padded = dim1_padded * dim2_padded;
    T * h = (T *)malloc(nbEle_padded * sizeof(T));
    T * h2 = (T *)malloc(nbEle_padded * sizeof(T));
    unsigned char * compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    HeatDis2D heatdis(temp_info.src_temp, temp_info.wall_temp, temp_info.ratio, size.dim1, size.dim2);
    heatdis.initData(h, h2, temp_info.init_temp);
    SZp_compress_2dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
    T * tmp = nullptr;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        clock_gettime(CLOCK_REALTIME, &start);
        SZp_decompress_2dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
            if(record){
                std::string h_name = heatdis2d_data_dir + "/h.L2.doc." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, nbEle_padded);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdis.reset_source(h, h2);
        heatdis.iterate(h, h2, tmp);
        SZp_compress_2dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
            printf("doc iter %d: cr = %.2f\n", iter-1, 1.0 * nbEle_padded * sizeof(T) / cmpSize);
            printf("doc iter %d: time = %.6f\n", iter-1, elapsed_time);
            fflush(stdout);
        }
    }
    {
        SZp_decompress_2dLorenzo(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound);
        std::string h_name = heatdis2d_data_dir + "/h.L2.doc." + std::to_string(iter);
        writefile(h_name.c_str(), h, nbEle_padded);
        printf("doc exit cr = %.2f\n", 1.0 * nbEle_padded * sizeof(T) / cmpSize);
    }
    printf("doc elapsed_time = %.6f\n", elapsed_time);
    free(compressed);
    free(h);
    free(h2);
}

template <class T>
void SZp_heatdis_2dLorenzo(
    unsigned char *cmpDataBuffer,
    size_t dim1, size_t dim2,
    int blockSideLength, int max_iter,
    float source_temp, float wall_temp,
    float init_temp, double ratio,
    double errorBound,
    decmpState state, bool record
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t dim1_padded = size.dim1 + 2;
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    size_t nbEle_padded = dim1_padded * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    int * Buffer_1d = (int *)malloc(buffer_dim2 * 5 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * signPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));

    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        cmpData[i] = (unsigned char *)malloc(nbEle_padded * sizeof(T));
        offsets[i] = (int *)malloc(size.block_dim1 * sizeof(int));
    }
    memcpy(cmpData[0], cmpDataBuffer, size.nbEle * sizeof(T));

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
            int fixed_rate = (int)cmpDataBuffer[block_index++];
            size_t savedbitsbytelength = compute_encoding_byteLength(block_size, fixed_rate);
            if(fixed_rate)
                prefix_length += (cmp_block_sign_length + savedbitsbytelength);
        }
    }
    TempInfo2D temp_info(source_temp, wall_temp, init_temp, ratio, errorBound);
    SZpAppBufferSet_2d<int> * buffer_set = new SZpAppBufferSet_2d<int>(buffer_dim1, buffer_dim2, Buffer_2d, Buffer_1d, appType::HEATDIS);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, offsets, absPredError, signPredError, signFlag);

    switch(state){
        case decmpState::postPred:{
            // temp_info.prepare_src_row(size.dim2, buffer_set->decmp_buffer, buffer_set->lorenzo_buffer);
            // heatdisUpdatePostPred<T>(size, cmpkit_set, buffer_set, temp_info, errorBound, max_iter, record);
            break;
        }
        case decmpState::prePred:{
            heatdisUpdatePrePred<T>(size, cmpkit_set, buffer_set, temp_info, errorBound, max_iter, record);
            break;
        }
        case decmpState::full:{
            heatdisUpdateDOC<T>(size, dim1_padded, buffer_dim2, temp_info, errorBound, max_iter, record);
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

template <class T>
void SZp_heatdis_2dLorenzo(
    unsigned char *cmpDataBuffer, ht2DSettings& s, decmpState state, bool record
){
    SZp_heatdis_2dLorenzo<T>(cmpDataBuffer, s.dim1, s.dim2, s.B, s.steps, s.src_temp, s.wall_temp, s.init_temp, s.ratio, s.eb, state, record);
}

#endif