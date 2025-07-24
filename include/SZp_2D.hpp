#ifndef _SZP_LORENZO_PREDICTOR_2D_HPP
#define _SZP_LORENZO_PREDICTOR_2D_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <array>
#include "typemanager.hpp"
#include "SZ_app_utils.hpp"
#include "utils.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZp_compress_fast(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * col_buffer = (int *)calloc(size.Bsize, sizeof(int));
    int * prevRow_buffer = (int *)calloc((size.dim2+1), sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        const T * y_data_pos = x_data_pos;
        int * prevRow = prevRow_buffer;
        memset(col_buffer, 0, size.Bsize*sizeof(int));
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            const T * curr_data_pos = y_data_pos;
            int max_err = 0;
            for(int i=0; i<size_x; i++){
                int prevLeft = col_buffer[i];
                for(int j=0; j<size_y; j++){
                    int q = SZ_quantize(*curr_data_pos, inver_eb);
                    int err_l = q - prevLeft;
                    int derr = err_l - prevRow[j];
                    prevRow[j] = err_l;
                    prevLeft = q;
                    if(j == size_y-1) col_buffer[i] = q;
                    (*sign_pos++) = (derr < 0);
                    unsigned u = abs(derr);
                    (*abs_err_pos++) = u;
                    max_err = max_err > u ? max_err : u;
                    curr_data_pos++;
                }
                curr_data_pos += size.offset_0 - size_y;
            }
            prevRow += size_y; 
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
        x_data_pos += size.Bsize * size.offset_0;
    }
    cmpSize = cmpData_pos - cmpData;
    free(col_buffer);
    free(prevRow_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_compress(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + offset_0 + 1;
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
                    int err = predict_lorenzo_2d(curr_data_pos++, curr_buffer_pos++, offset_0, inver_eb);
                    (*sign_pos++) = (err < 0);
                    unsigned int abs_err = abs(err);
                    (*abs_err_pos++) = abs_err;
                    max_err = max_err > abs_err ? max_err : abs_err;
                }
                block_buffer_pos += offset_0;
                curr_data_pos += size.offset_0 - size_y;
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
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
    cmpSize = cmpData_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_decompress_fast(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    double twice_eb = 2 * errorBound;
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * colPrefix = (int *)malloc(size.dim2*sizeof(int));
    memset(colPrefix, 0, size.dim2*sizeof(int));
    int block_ind = 0;
    size_t x, y;
    for(x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        int * buffer_start_pos = quant_buffer + offset_0 + 1;
        for(y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            int index, curr;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                index = 0;
                for(int i=0; i<size_x; i++){
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr_buffer_pos[j] = 0 - absPredError[index];
                        // else curr_buffer_pos[j] = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr_buffer_pos[j] = (absPredError[index] ^ s) - s;
                        index++;
                    }
                    block_buffer_pos += offset_0;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int j=0; j<size_y; j++){
                        curr_buffer_pos[j] = 0;
                    }
                    block_buffer_pos += offset_0;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        int * curr_pos = quant_buffer + offset_0 + 1;
        for(int i=0; i<size_x; i++){
            T * curr_data_pos = decData + (x * size.Bsize + i) * size.offset_0;
            int rowPrefix = 0;
            for(size_t j=0; j<size.dim2; j++){
                rowPrefix += curr_pos[j];
                colPrefix[j] += rowPrefix;
                curr_data_pos[j] = colPrefix[j] * twice_eb;
            }
            curr_pos += offset_0;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
    free(colPrefix);
}

template <class T>
void SZp_decompress(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    double twice_eb = errorBound * 2;
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            T * curr_data_pos = y_data_pos;
            if(!fixed_rate){
                memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
            }
            int index = 0;
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    int s = -(int)signFlag[index];
                    curr_buffer_pos[0] = (absPredError[index] ^ s) - s;
                    index++;
                    recover_lorenzo_2d(curr_buffer_pos, offset_0);
                    curr_data_pos[0] = curr_buffer_pos[0] * twice_eb;
                    curr_data_pos++;
                    curr_buffer_pos++;
                }
                block_buffer_pos += offset_0;
                curr_data_pos += size.offset_0 - size_y;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.dim2;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_decompress_postPred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            int * curr_data_pos = y_data_pos;
            if(!fixed_rate){
                memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
            }
            int index = 0;
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    // if(signFlag[index]) curr_buffer_pos[j] = 0 - absPredError[index];
                    // else curr_buffer_pos[j] = absPredError[index];
                    int s = -(int)signFlag[index];
                    curr_buffer_pos[j] = (absPredError[index] ^ s) - s;
                    index++;
                }
                block_buffer_pos += offset_0;
                curr_data_pos += size.offset_0 - size_y;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.dim2;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_decompress_prePred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            int * curr_data_pos = y_data_pos;
            if(!fixed_rate){
                memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
            }
            int index = 0;
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    int s = -(int)signFlag[index];
                    curr_buffer_pos[0] = (absPredError[index] ^ s) - s;
                    index++;
                    recover_lorenzo_2d(curr_buffer_pos, offset_0);
                    curr_data_pos[0] = curr_buffer_pos[0];
                    curr_data_pos++;
                    curr_buffer_pos++;
                }
                block_buffer_pos += offset_0;
                curr_data_pos += size.offset_0 - size_y;
            }
            buffer_start_pos += size.Bsize;
            y_data_pos += size.Bsize;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.dim2;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

double SZp_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                int curr;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr = 0 - absPredError[index];
                        // else curr = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr = (absPredError[index] ^ s) - s;
                        index++;
                        quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * curr;
                    }
                }
            }
            index_y += size.Bsize;
        }
        index_x += size.Bsize;
    }
    free(absPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

double SZp_mean_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + offset_0 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(!fixed_rate){
                memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
            }
            int * curr_buffer_pos = block_buffer_pos;
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    // if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                    // else curr_buffer_pos[0] = absPredError[index];
                    int s = -(int)signFlag[index];
                    curr_buffer_pos[0] = (absPredError[index] ^ s) - s;
                    index++;
                    recover_lorenzo_2d(curr_buffer_pos, offset_0);
                    quant_sum += curr_buffer_pos[0];
                    curr_buffer_pos++;
                }
                curr_buffer_pos += offset_0 - size_y;
            }
            buffer_start_pos += size.Bsize;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            mean = SZp_mean_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZp_stddev_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int64_t * colPrefix = (int64_t *)malloc(size.dim2*sizeof(int64_t));
    memset(colPrefix, 0, size.dim2*sizeof(int64_t));
    int block_ind = 0;
    int64_t d;
    int64_t quant_sum = 0;
    uint64_t d2;
    uint64_t squared_quant_sum = 0;
    size_t x, y;
    for(x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        int * buffer_start_pos = quant_buffer + offset_0 + 1;
        for(y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            int index, curr;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                index = 0;
                for(int i=0; i<size_x; i++){
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int j=0; j<size_y; j++){
                        // if(signFlag[index]) curr_buffer_pos[j] = 0 - absPredError[index];
                        // else curr_buffer_pos[j] = absPredError[index];
                        int s = -(int)signFlag[index];
                        curr_buffer_pos[j] = (absPredError[index] ^ s) - s;
                        index++;
                    }
                    block_buffer_pos += offset_0;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int j=0; j<size_y; j++){
                        curr_buffer_pos[j] = 0;
                    }
                    block_buffer_pos += offset_0;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        int * curr_pos = quant_buffer + offset_0 + 1;
        for(int i=0; i<size_x; i++){
            int64_t rowPrefix = 0;
            for(size_t j=0; j<size.dim2; j++){
                rowPrefix += static_cast<int64_t>(curr_pos[j]);
                colPrefix[j] += rowPrefix;
                quant_sum += colPrefix[j];
                squared_quant_sum += static_cast<uint64_t>(colPrefix[j] *  colPrefix[j]);
            }
            curr_pos += offset_0;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
    free(colPrefix);
    double std = (2 * errorBound) * sqrt(((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1));
    return std;
}

double SZp_stddev_prePred(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t offset_0 = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + offset_0 + 1;
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                cmpData_pos += savedbitsbytelength;
            }else{
                memset(absPredError, 0, size.max_num_block_elements*sizeof(int));                
            }
            int * curr_buffer_pos = block_buffer_pos;
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    // if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                    // else curr_buffer_pos[0] = absPredError[index];
                    int s = -(int)signFlag[index];
                    curr_buffer_pos[0] = (absPredError[index] ^ s) - s;
                    index++;
                    recover_lorenzo_2d(curr_buffer_pos, offset_0);
                    int64_t d = static_cast<int64_t>(curr_buffer_pos[0]);
                    uint64_t d2 = d * d;
                    quant_sum += d;
                    squared_quant_sum += d2;
                    curr_buffer_pos++;
                }
                curr_buffer_pos += offset_0 - size_y;
            }
            buffer_start_pos += size.Bsize;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
    double std = (2 * errorBound) * sqrt(((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1));
    return std;
}

template <class T>
double SZp_stddev_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2;
    SZp_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double std = 0;
    for(size_t i=0; i<nbEle; i++) std += (decData[i] - mean) * (decData[i] - mean);
    std /= (nbEle - 1);
    return sqrt(std);
}

template <class T>
double SZp_stddev(
    unsigned char *cmpData, size_t dim1, size_t dim2, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double std;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            std = SZp_stddev_decOp(cmpData, dim1, dim2, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            std = SZp_stddev_prePred(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            std = SZp_stddev_postPred(cmpData, dim1, dim2, blockSideLength, errorBound);   
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return std;
}

inline void recoverBlockSlice2PostPred(
    size_t x, DSize_2d& size, unsigned char *cmpData,
    CmpBufferSet *cmpkit_set, unsigned char *& encode_pos,
    int *buffer_data_pos, size_t offset_0
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
                for(int j=0; j<size_y; j++){
                    curr_buffer_pos[j] = 0;
                }
                curr_buffer_pos += offset_0;
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    // if(cmpkit_set->signFlag[index]) curr_buffer_pos[j] = 0 - cmpkit_set->absPredError[index];
                    // else curr_buffer_pos[j] = cmpkit_set->absPredError[index];
                    int s = -(int)cmpkit_set->signFlag[index];
                    curr_buffer_pos[j] = (cmpkit_set->absPredError[index] ^ s) - s;
                    index++;
                }
                curr_buffer_pos += offset_0;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockSlice2PrePred(
    size_t x, DSize_2d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    AppBufferSet_2d *buffer_set, CmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
        if(fixed_rate){
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
            encode_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
            encode_pos += savedbitsbytelength;
        }
        else{
            for(int i=0; i<block_size; i++){
                cmpkit_set->absPredError[i] = 0;
            }
        }
        int * block_buffer_pos = buffer_start_pos;
        int index = 0;
        for(int i=0; i<size_x; i++){
            int * curr_buffer_pos = block_buffer_pos;
            for(int j=0; j<size_y; j++){
                // if(cmpkit_set->signFlag[index]) curr_buffer_pos[0] = 0 - cmpkit_set->absPredError[index];
                // else curr_buffer_pos[0] = cmpkit_set->absPredError[index];
                int s = -(int)cmpkit_set->signFlag[index];         // 0 or -1
                curr_buffer_pos[0] = (cmpkit_set->absPredError[index] ^ s) - s;
                index++;
                recover_lorenzo_2d(curr_buffer_pos, buffer_set->offset_0);
                curr_buffer_pos++;
            }
            block_buffer_pos += buffer_set->offset_0;
        }
        buffer_start_pos += size.Bsize;
    }
    memcpy(buffer_set->decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_set->offset_0-1, buffer_set->offset_0*sizeof(int));
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockSlicePostPred(
    size_t x, DSize_2d& size, AppBufferSet_2d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        int dx_buffer = 0;
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            dx_buffer += curr_row[j] + next_row[j];
            dx_pos[index] = dx_buffer * errorBound;
            buffer_set->dy_buffer[j] += curr_row[j] + curr_row[j+1];
            dy_pos[index] = buffer_set->dy_buffer[j] * errorBound;
            index++;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlockSlicePrePred(
    size_t x, DSize_2d& size, AppBufferSet_2d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * dx_pos = dx_start_pos;
    T * dy_pos = dy_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            dx_pos[index] = (next_row[j] - prev_row[j]) * errorBound;
            dy_pos[index] = (curr_row[j+1] - curr_row[j-1]) * errorBound;
            index++;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdyProcessBlocksPostPred(
    DSize_2d& size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            dxdyProcessBlockSlicePostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }else if(x == size.block_dim1 - 1){
            buffer_set->currSlice_data_pos = buffer_set->nextSlice_data_pos;
            dxdyProcessBlockSlicePostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
        }else{
            std::swap(buffer_set->currSlice_data_pos, buffer_set->nextSlice_data_pos);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            dxdyProcessBlockSlicePostPred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdyProcessBlocksPrePred(
    DSize_2d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, encode_pos, buffer_set->currSlice_data_pos, buffer_set, cmpkit_set);
            memcpy(buffer_set->nextSlice_data_pos - buffer_set->offset_0 - 1, buffer_set->decmp_buffer, buffer_set->offset_0 * sizeof(int));
            recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
            dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempSlice_pos);
            if(x == size.block_dim1 - 1){
                dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextSlice_data_pos - buffer_set->offset_0 - 1, buffer_set->decmp_buffer, buffer_set->offset_0 * sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
                dxdyProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_dxdy(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    AppBufferSet_2d * buffer_set = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
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
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(absPredError);
    free(signFlag);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize_2d& size,
    AppBufferSet_2d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    T * result_pos = result_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        const int * prev_row = curr_row - buffer_set->offset_0;
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            result_pos[index++] = (curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j]) * errorBound * 2;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlockSlicePostPred(
    size_t x, DSize_2d& size,
    AppBufferSet_2d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    const int * prevBlockSliceBottom_pos = isTopSlice ? nullptr : buffer_set->prevSlice_data_pos + (size.Bsize - 1) * buffer_set->offset_0 - 1;
    const int * nextBlockSliceTop_pos = isBottomSlice ? nullptr : buffer_set->nextSlice_data_pos - 1;
    if(!isTopSlice) memcpy(buffer_set->currSlice_data_pos-buffer_set->offset_0-1, prevBlockSliceBottom_pos, buffer_set->offset_0*sizeof(int));
    if(!isBottomSlice) memcpy(buffer_set->currSlice_data_pos+size.Bsize*buffer_set->offset_0-1, nextBlockSliceTop_pos, buffer_set->offset_0*sizeof(int));
    T * result_pos = result_start_pos;
    const int * curr_row = buffer_set->currSlice_data_pos;
    int index = 0;
    for(int i=0; i<size_x; i++){
        int x_buffer = 0, x_buffer_2 = 0;
        buffer_set->dy_buffer[0] += curr_row[0];
        const int * next_row = curr_row + buffer_set->offset_0;
        for(size_t j=0; j<size.dim2; j++){
            buffer_set->dy_buffer[j+1] += curr_row[j+1];
            x_buffer += curr_row[j];
            x_buffer_2 += next_row[j];
            result_pos[index++] = (buffer_set->dy_buffer[j+1] - buffer_set->dy_buffer[j] + x_buffer_2 - x_buffer) * errorBound * 2;
        }
        curr_row += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize_2d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, encode_pos, buffer_set->currSlice_data_pos, buffer_set, cmpkit_set);
            memcpy(buffer_set->nextSlice_data_pos - buffer_set->offset_0 - 1, buffer_set->decmp_buffer, buffer_set->offset_0 * sizeof(int));
            recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
            laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempSlice_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                memcpy(buffer_set->nextSlice_data_pos - buffer_set->offset_0 - 1, buffer_set->decmp_buffer, buffer_set->offset_0 * sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void laplacianProcessBlocksPostPred(
    DSize_2d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_2d *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2;
    buffer_set->reset();
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->currSlice_data_pos, buffer_set->offset_0);
            recoverBlockSlice2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
            laplacianProcessBlockSlicePostPred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempSlice_pos);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set->compressed, cmpkit_set, encode_pos, buffer_set->nextSlice_data_pos, buffer_set->offset_0);
                laplacianProcessBlockSlicePostPred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_laplacian(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    AppBufferSet_2d * buffer_set = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * laplacian_pos = laplacian_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            laplacianProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress(decData, cmpData, dim1, dim2, blockSideLength, errorBound);
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
    free(absPredError);
    free(signFlag);
    free(decData);
}

// divergence
template <class T>
inline void divergenceProcessBlockSlicePrePred(
    size_t x, DSize_2d& size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int * vx_row_prev = vx_curr_row - off_0;
        const int * vx_row_next = vx_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            int dfxx = vx_row_next[j] - vx_row_prev[j];
            int dfyy = vy_curr_row[j+1] - vy_curr_row[j-1];
            divergence_pos[index++] = (dfxx + dfyy) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlockSlicePostPred(
    size_t x, DSize_2d& size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        int vx_dx_buffer = 0;
        const int * vx_row_next = vx_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            vx_dx_buffer += vx_curr_row[j] + vx_row_next[j];
            buffer_set[1]->dy_buffer[j] += vy_curr_row[j] + vy_curr_row[j+1];
            divergence_pos[index++] = (vx_dx_buffer + buffer_set[1]->dy_buffer[j]) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlocksPrePred(
    DSize_2d &size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, encode_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                memcpy(buffer_set[i]->nextSlice_data_pos-off_0-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
            }
            divergenceProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.block_dim1 - 1){
                divergenceProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                for(i=0; i<2; i++){
                    memcpy(buffer_set[i]->nextSlice_data_pos-off_0-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                    recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                }
                divergenceProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void divergenceProcessBlocksPostPred(
    DSize_2d &size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            divergenceProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.block_dim1 - 1){
                divergenceProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                divergenceProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_divergence(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet_2d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            divergenceProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            divergenceProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZp_decompress(decData[i], cmpData[i], dim1, dim2, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_divergence_2d(dim1, dim2, decData[0], decData[1], divergence_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<2; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_2d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}

// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize_2d& size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curl_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int * vy_row_prev = vy_curr_row - off_0;
        const int * vy_row_next = vy_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            curl_pos[index++] = ((vy_row_next[j] - vy_row_prev[j]) - (vx_curr_row[j+1] - vx_curr_row[j-1])) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlockSlicePostPred(
    size_t x, DSize_2d& size, size_t off_0,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curl_pos = result_start_pos;
    const int * vx_curr_row = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_row = buffer_set[1]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        int vy_dx_buffer = 0;
        const int * vy_row_next = vy_curr_row + off_0;
        for(size_t j=0; j<size.dim2; j++){
            vy_dx_buffer += vy_curr_row[j] + vy_row_next[j];
            buffer_set[0]->dy_buffer[j] += vx_curr_row[j] + vx_curr_row[j+1];
            curl_pos[index++] = (vy_dx_buffer - buffer_set[0]->dy_buffer[j]) * errorBound;
        }
        vx_curr_row += off_0;
        vy_curr_row += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlocksPrePred(
    DSize_2d &size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PrePred(x, size, encode_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                memcpy(buffer_set[i]->nextSlice_data_pos-off_0-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
            }
            curlProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                for(i=0; i<2; i++){
                    memcpy(buffer_set[i]->nextSlice_data_pos-off_0-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                    recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                }
                curlProcessBlockSlicePrePred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void curlProcessBlocksPostPred(
    DSize_2d &size,
    std::array<CmpBufferSet *, 2>& cmpkit_set,
    std::array<AppBufferSet_2d *, 2>& buffer_set,
    std::array<unsigned char *, 2>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2;
    for(i=0; i<2; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    int * tempSlice_pos = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<2; i++){
                recoverBlockSlice2PostPred(x, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->currSlice_data_pos, off_0);
                recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
            }
            curlProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, true, false);
        }
        else{
            for(i=0; i<2; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempSlice_pos);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, true);
            }
            else{
                for(i=0; i<2; i++){
                    recoverBlockSlice2PostPred(x+1, size, cmpkit_set[i]->compressed, cmpkit_set[i], encode_pos[i], buffer_set[i]->nextSlice_data_pos, off_0);
                }
                curlProcessBlockSlicePostPred(x, size, off_0, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_curl(
    std::array<unsigned char *, 2> cmpData,
    size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *curl_result, decmpState state
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    std::array<int *, 2> Buffer_2d = {nullptr, nullptr};
    std::array<unsigned int *, 2> absPredError = {nullptr, nullptr};
    std::array<T *, 2> decData = {nullptr, nullptr};
    std::array<unsigned char *, 2> signFlag = {nullptr, nullptr};
    std::array<AppBufferSet_2d *, 2> buffer_set = {nullptr, nullptr};
    std::array<CmpBufferSet *, 2> cmpkit_set = {nullptr, nullptr};
    std::array<unsigned char *, 2> encode_pos = {nullptr, nullptr};
    for(int i=0; i<2; i++){
        Buffer_2d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * curl_pos = curl_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            curlProcessBlocksPostPred(size, cmpkit_set, buffer_set, encode_pos, curl_pos, errorBound);
            break;
        }
        case decmpState::prePred:{
            curlProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, curl_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<2; i++){
                SZp_decompress(decData[i], cmpData[i], dim1, dim2, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_curl_2d(dim1, dim2, decData[0], decData[1], curl_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }

    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<2; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_2d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}

#endif
