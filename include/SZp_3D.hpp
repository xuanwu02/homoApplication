#ifndef _SZP_LORENZO_PREDICTOR_3D_HPP
#define _SZP_LORENZO_PREDICTOR_3D_HPP

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
void SZp_compress(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            const T * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                unsigned int * abs_diff_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                int quant_diff, max_quant_diff = 0;
                const T * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            quant_diff = predict_lorenzo_3d(curr_data_pos++, curr_buffer_pos++, inver_eb, offset_0, offset_1);
                            (*sign_pos++) = (quant_diff < 0);
                            unsigned int abs_diff = abs(quant_diff);
                            (*abs_diff_pos++) = abs_diff;
                            max_quant_diff = max_quant_diff > abs_diff ? max_quant_diff : abs_diff;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
                int fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                    cmpData_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, cmpData_pos, fixed_rate);
                    cmpData_pos += savedbitsbyteLength;
                }
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
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
void SZp_decompress(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    double twice_eb = errorBound * 2;
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            T * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(!fixed_rate){
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(unsigned int));
                }else{
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }
                T * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                            else curr_buffer_pos[0] = absPredError[index];
                            index++;
                            recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
                            curr_data_pos[0] = curr_buffer_pos[0] * twice_eb;
                            curr_data_pos++;
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_decompress_postPred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(!fixed_rate){
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
                }else{
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }
                int * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(signFlag[index]) curr_buffer_pos[k] = 0 - absPredError[index];
                            else curr_buffer_pos[k] = absPredError[index];
                            index++;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

void SZp_decompress_prePred(
    int *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
// clock_gettime(CLOCK_REALTIME, &start2);
    for(size_t x=0; x<size.block_dim1; x++){
        int * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(!fixed_rate){
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
                }else{
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }
                int * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                            else curr_buffer_pos[0] = absPredError[index];
                            index++;
                            recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
                            curr_data_pos[0] = curr_buffer_pos[0];
                            curr_data_pos++;
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += offset_1;
                        curr_data_pos += size.offset_1 - size_z;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                    curr_data_pos += size.offset_0 - size_y * size.offset_1;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.offset_1;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
        x_data_pos += size.Bsize * size.offset_0;
    }
// clock_gettime(CLOCK_REALTIME, &end2);
// rec_time = get_elapsed_time(start2, end2);
    free(pred_buffer);
    free(absPredError);
    free(signFlag);
}

double SZp_mean_postPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    int index_x = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int index_y = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            int index_z = 0;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                    int index = 0;
                    int curr;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                if(signFlag[index]) curr = 0 - absPredError[index];
                                else curr = absPredError[index];
                                index++;
                                quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (size.dim3 - (index_z + k)) * curr;
                            }
                        }
                    }
                }
                index_z += size.Bsize;
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
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }else{
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
                }
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                            else curr_buffer_pos[0] = absPredError[index];
                            index++;
                            recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
                            quant_sum += curr_buffer_pos[0];
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += offset_1;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(absPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean_postPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            mean = SZp_mean_prePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

// double SZp_variance_postPredMean(
//     unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
//     int blockSideLength, double errorBound
// ){
//     DSize_3d size(dim1, dim2, dim3, blockSideLength);
//     size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
//     size_t offset_1 = size.dim3 + 1;
//     int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
//     memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
//     unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
//     unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
//     unsigned char * cmpData_pos = cmpData + size.num_blocks;
//     int block_ind = 0;
//     int64_t quant_sum = 0;
//     uint64_t squared_quant_sum = 0;
//     int index_x = 0;
//     for(size_t x=0; x<size.block_dim1; x++){
//         int index_y = 0;
//         int * buffer_start_pos = quant_buffer + offset_0 + offset_1 + 1;
//         for(size_t y=0; y<size.block_dim2; y++){
//             int index_z = 0;
//             for(size_t z=0; z<size.block_dim3; z++){
//                 int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
//                 int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
//                 int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
//                 int block_size = size_x * size_y * size_z;
//                 int fixed_rate = (int)cmpData[block_ind++];
//                 if(fixed_rate){
//                     size_t cmp_block_sign_length = (block_size + 7) / 8;
//                     convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
//                     cmpData_pos += cmp_block_sign_length;
//                     unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
//                     cmpData_pos += savedbitsbytelength;
//                     int index = 0;
//                     int curr;
//                     for(int i=0; i<size_x; i++){
//                         for(int j=0; j<size_y; j++){
//                             for(int k=0; k<size_z; k++){
//                                 if(signFlag[index]) curr = 0 - absPredError[index];
//                                 else curr = absPredError[index];
//                                 index++;
//                                 quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (size.dim3 - (index_z + k)) * curr;
//                             }
//                         }
//                     }
//                 }else{
//                     memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
//                 }
//                 int * block_buffer_pos = buffer_start_pos;
//                 for(int i=0; i<size_x; i++){
//                     for(int j=0; j<size_y; j++){
//                         memcpy(block_buffer_pos, absPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
//                         int * curr_buffer_pos = block_buffer_pos;
//                         for(int k=0; k<size_z; k++){
//                             recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
//                             int64_t d = static_cast<int64_t>(curr_buffer_pos[0]);
//                             uint64_t d2 = d * d;
//                             squared_quant_sum += d2;
//                             curr_buffer_pos++;
//                         }
//                         block_buffer_pos += offset_1;
//                     }
//                     block_buffer_pos += offset_0 - size_y * offset_1;
//                 }
//                 buffer_start_pos += size.Bsize;
//                 index_z += size.Bsize;
//             }
//             buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
//             index_y += size.Bsize;
//         }
//         memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
//         index_x += size.Bsize;
//     }
//     free(quant_buffer);
//     free(absPredError);
//     free(signFlag);
//     double var = (2 * errorBound) * (2 * errorBound) * ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
//     return var;
// }

double SZp_variance_prePredMean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t offset_0 = (size.dim2 + 1) * (size.dim3 + 1);
    size_t offset_1 = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + offset_0 + offset_1 + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                    cmpData_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, absPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                }else{
                    memset(absPredError, 0, size.max_num_block_elements*sizeof(int));
                }
                int * block_buffer_pos = buffer_start_pos;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(signFlag[index]) curr_buffer_pos[0] = 0 - absPredError[index];
                            else curr_buffer_pos[0] = absPredError[index];
                            index++;
                            recover_lorenzo_3d(curr_buffer_pos, offset_0, offset_1);
                            int64_t d = static_cast<int64_t>(curr_buffer_pos[0]);
                            uint64_t d2 = d * d;
                            quant_sum += d;
                            squared_quant_sum += d2;
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += offset_1;
                    }
                    block_buffer_pos += offset_0 - size_y * offset_1;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * offset_1 - size.Bsize * size.block_dim3;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*offset_0, offset_0*sizeof(int));
    }
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
    double var = (2 * errorBound) * (2 * errorBound) * ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
    return var;
}

template <class T>
double SZp_variance_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZp_variance(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZp_variance_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZp_variance_prePredMean(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            // var = SZp_variance_postPredMean(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockSlice2PostPred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    AppBufferSet_3d *buffer_set, CmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * block_buffer_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                int index = 0;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            if(cmpkit_set->signFlag[index]) curr_buffer_pos[k] = 0 - cmpkit_set->absPredError[index];
                            else curr_buffer_pos[k] = cmpkit_set->absPredError[index];
                            index++;
                        }
                        block_buffer_pos += buffer_set->offset_1;
                    }
                    block_buffer_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                }
            }else{
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            curr_buffer_pos[k] = 0;
                        }
                        block_buffer_pos += buffer_set->offset_1;
                    }
                    block_buffer_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->offset_1 - size.Bsize * size.block_dim3;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

inline void recoverBlockSlice2PrePred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    AppBufferSet_3d *buffer_set, CmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * curr_buffer_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->absPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
            }else{
                for(int i=0; i<block_size; i++) cmpkit_set->absPredError[i] = 0;
            }
            int index = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        if(cmpkit_set->signFlag[index]) curr_buffer_pos[0] = 0 - cmpkit_set->absPredError[index];
                        else curr_buffer_pos[0] = cmpkit_set->absPredError[index];
                        index++;
                        recover_lorenzo_3d(curr_buffer_pos++, buffer_set->offset_0, buffer_set->offset_1);
                    }
                    curr_buffer_pos += buffer_set->offset_1 - size_z;
                }
                curr_buffer_pos += buffer_set->offset_0 - size_y * buffer_set->offset_1;
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->offset_1 - size.Bsize * size.block_dim3;
    }
    memcpy(buffer_set->decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_set->offset_0-buffer_set->offset_1-1, buffer_set->offset_0*sizeof(int));
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlockSlicePostPred(
    size_t x, DSize_3d size, derivIntBuffer_3d *deriv_buffer,
    AppBufferSet_3d *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t dx_offset_0 = size.dim3 + 1;
    size_t dy_offset_0 = size.dim3 + 1;
    size_t dz_offset_0 = size.dim2 + 1;
    for(int i=0; i<size_x; i++){
        size_t global_x_offset = x * size.Bsize + i;
        T * dx_pos = dx_start_pos + i * size.offset_0;
        T * dy_pos = dy_start_pos + i * size.offset_0;
        T * dz_pos = dz_start_pos + i * size.offset_0;
        const int * dx_level_0_pos = buffer_set->currSlice_data_pos + buffer_set->offset_0 * i;
        const int * dx_level_1_pos = buffer_set->currSlice_data_pos + buffer_set->offset_0 * (i + 1);
        const int * curr_x_plane = buffer_set->currSlice_data_pos + buffer_set->offset_0 * i;
        for(size_t j=0; j<size.dim2; j++){
            int * dx_int_buffer_pos = deriv_buffer->dx_buffer + (j + 1) * dx_offset_0 + 1; 
            const int * dy_level_0_pos = curr_x_plane + j * buffer_set->offset_1;
            const int * dy_level_1_pos = curr_x_plane + (j + 1) * buffer_set->offset_1;
            int * dy_int_buffer_pos = deriv_buffer->dy_buffer[j] + dy_offset_0 * (global_x_offset + 1) + 1;
            const int * curr_y_row = curr_x_plane + j * buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                const int * dz_level_0_pos = curr_y_row + k;
                const int * dz_level_1_pos = curr_y_row + k + 1;
                int * dz_int_buffer_pos = deriv_buffer->dz_buffer[k] + dz_offset_0 * (global_x_offset + 1) + j + 1;
                deriv_lorenzo_2d(dx_level_0_pos++, dx_level_1_pos++, dx_int_buffer_pos++, dx_pos++, dx_offset_0, errorBound);
                deriv_lorenzo_2d(dy_level_0_pos++, dy_level_1_pos++, dy_int_buffer_pos++, dy_pos++, dy_offset_0, errorBound);
                deriv_lorenzo_2d(dz_level_0_pos, dz_level_1_pos, dz_int_buffer_pos, dz_pos++, dz_offset_0, errorBound);
            }
            dx_level_0_pos += 2;
            dx_level_1_pos += 2;
        }
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlocksPostPred(
    DSize_3d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    derivIntBuffer_3d *deriv_buffer,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PostPred(x, size, encode_pos, buffer_set->currSlice_data_pos, buffer_set, cmpkit_set);
            recoverBlockSlice2PostPred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
            dxdydzProcessBlockSlicePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            std::swap(buffer_set->currSlice_data_pos, buffer_set->nextSlice_data_pos);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockSlicePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockSlice2PostPred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
                dxdydzProcessBlockSlicePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
inline void dxdydzProcessBlockSlicePrePred(
    size_t x, DSize_3d& size, AppBufferSet_3d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * curr_plane = buffer_set->currSlice_data_pos - buffer_set->offset_1 - 1;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.offset_0;
        T * x_dy_pos = dy_start_pos + i * size.offset_0;
        T * x_dz_pos = dz_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane + buffer_set->offset_1 + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.offset_1;
            T * y_dy_pos = x_dy_pos + j * size.offset_1;
            T * y_dz_pos = x_dz_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index_2d = (j + 1) * buffer_set->offset_1 + k + 1;
                y_dx_pos[k] = (next_plane[buffer_index_2d] - prev_plane[buffer_index_2d]) * errorBound;
                y_dy_pos[k] = (next_row[k] - prev_row[k]) * errorBound;
                y_dz_pos[k] = (curr_row[k + 1] - curr_row[k - 1]) * errorBound;
            }
            curr_row += buffer_set->offset_1;
        }
        curr_plane += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, encode_pos, buffer_set->currSlice_data_pos, buffer_set, cmpkit_set);
            memcpy(buffer_set->nextSlice_data_pos-buffer_set->offset_0-buffer_set->offset_1-1, buffer_set->decmp_buffer, buffer_set->offset_0*sizeof(int));
            recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
            dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, true);
            }else{
                memcpy(buffer_set->nextSlice_data_pos-buffer_set->offset_0-buffer_set->offset_1-1, buffer_set->decmp_buffer, buffer_set->offset_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
                dxdydzProcessBlockSlicePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_dxdydz(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    size_t dim3, int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, T *dz_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    AppBufferSet_3d * buffer_set = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            int * dx_buffer = allocateAndZero1D((size.dim2+1) * (size.dim3+1));
            int** dy_buffer = allocateAndZero2D(size.dim2, (size.dim1+1) * (size.dim3+1));
            int** dz_buffer = allocateAndZero2D(size.dim3, (size.dim1+1) * (size.dim2+1));
            derivIntBuffer_3d * deriv_buffer = new derivIntBuffer_3d(dx_buffer, dy_buffer, dz_buffer);
            dxdydzProcessBlocksPostPred(size, cmpkit_set, buffer_set, deriv_buffer, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            for(size_t i=0; i<std::max(size.dim2, size.dim3); i++) {
                if(i<size.dim2) delete[] dy_buffer[i];
                if(i<size.dim3) delete[] dz_buffer[i];
            }
            delete[] dy_buffer;
            delete[] dz_buffer;
            delete[] dx_buffer;
            delete deriv_buffer;
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(absPredError);
    free(signFlag);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockSlicePrePred(
    size_t x, DSize_3d& size, AppBufferSet_3d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->setGhostEle(size, isTopSlice, isBottomSlice);
    const int * curr_plane = buffer_set->currSlice_data_pos - buffer_set->offset_1 - 1;
    for(int i=0; i<size_x; i++){
        T * laplacian_pos = result_start_pos + i * size.offset_0;
        const int * prev_plane = curr_plane - buffer_set->offset_0;
        const int * next_plane = curr_plane + buffer_set->offset_0;
        const int * curr_row = curr_plane + buffer_set->offset_1 + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_laplacian_pos = laplacian_pos + j * size.offset_1;
            const int * prev_row = curr_row - buffer_set->offset_1;
            const int * next_row = curr_row + buffer_set->offset_1;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                size_t buffer_index_2d = (j + 1) * buffer_set->offset_1 + k + 1;
                y_laplacian_pos[index_1d] = (curr_row[k-1] + curr_row[k+1] +
                                             prev_row[k] + next_row[k] +
                                             prev_plane[buffer_index_2d] + next_plane[buffer_index_2d] -
                                             6 * curr_row[k]) * errorBound * 2;
            }
            curr_row += buffer_set->offset_1;
        }
        curr_plane += buffer_set->offset_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize_3d &size,
    CmpBufferSet *cmpkit_set, 
    AppBufferSet_3d *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            recoverBlockSlice2PrePred(x, size, encode_pos, buffer_set->currSlice_data_pos, buffer_set, cmpkit_set);
            memcpy(buffer_set->nextSlice_data_pos-buffer_set->offset_0-buffer_set->offset_1-1, buffer_set->decmp_buffer, buffer_set->offset_0*sizeof(int));
            recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
            laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currSlice_data_pos, buffer_set->prevSlice_data_pos, buffer_set->nextSlice_data_pos, tempBlockSlice);
            if(x == size.block_dim1 - 1){
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                memcpy(buffer_set->nextSlice_data_pos-buffer_set->offset_0-buffer_set->offset_1-1, buffer_set->decmp_buffer, buffer_set->offset_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos, buffer_set->nextSlice_data_pos, buffer_set, cmpkit_set);
                laplacianProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_laplacian(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    size_t dim3, int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    AppBufferSet_3d * buffer_set = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d);
    CmpBufferSet * cmpkit_set = new CmpBufferSet(cmpData, absPredError, signFlag, nullptr);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
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
            laplacianProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(absPredError);
    free(signFlag);
    free(decData);
}

// divergence
template <class T>
inline void divergenceProcessBlockSlicePrePred(
    size_t x, DSize_3d& size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    T *result_start_pos, double errorBound,
    size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set[0]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * divergence_pos = result_start_pos;
    const int * vx_curr_plane = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_plane = buffer_set[1]->currSlice_data_pos;
    const int * vz_curr_plane = buffer_set[2]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        const int* vx_plane_prev = vx_curr_plane - off_0;
        const int* vx_plane_next = vx_curr_plane + off_0;
        const int* vy_plane_base = vy_curr_plane;
        const int* vz_plane_base = vz_curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            size_t row_offset = j * off_1;
            const int* vx_row_prev = vx_plane_prev + row_offset;
            const int* vx_row_next = vx_plane_next + row_offset;
            const int* vy_row_prev = vy_plane_base - off_1;
            const int* vy_row_next = vy_plane_base + off_1;
            const int* vz_row = vz_plane_base;
            for(size_t k=0; k<size.dim3; k++){
                int dfxx = vx_row_next[k] - vx_row_prev[k];
                int dfyy = vy_row_next[k] - vy_row_prev[k];
                int dfzz = vz_row[k+1] - vz_row[k-1];
                divergence_pos[index++] = (dfxx + dfyy + dfzz) * errorBound;
            }
            vy_plane_base += off_1;
            vz_plane_base += off_1;
        }
        vx_curr_plane += off_0;
        vy_curr_plane += off_0;
        vz_curr_plane += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void divergenceProcessBlocksPrePred(
    DSize_3d &size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    T *result_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, encode_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                memcpy(buffer_set[i]->nextSlice_data_pos-off_0-off_1-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
            }
            divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    memcpy(buffer_set[i]->nextSlice_data_pos-off_0-off_1-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                    recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                }
                divergenceProcessBlockSlicePrePred(x, size, buffer_set, result_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_divergence(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *divergence_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet_3d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc((buffer_size * 4) * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * divergence_pos = divergence_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            divergenceProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, divergence_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<3; i++){
                SZp_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_divergence_3d(dim1, dim2, dim3, decData[0], decData[1], decData[2], divergence_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<3; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_3d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}

// curl
template <class T>
inline void curlProcessBlockSlicePrePred(
    size_t x, DSize_3d& size,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    T *curlx_start_pos, T *curly_start_pos, T *curlz_start_pos,
    double errorBound, size_t off_0, size_t off_1,
    bool isTopSlice, bool isBottomSlice
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set[1]->setGhostEle(size, isTopSlice, isBottomSlice);
    buffer_set[2]->setGhostEle(size, isTopSlice, isBottomSlice);
    size_t index = 0;
    T * curlx_pos = curlx_start_pos;
    T * curly_pos = curly_start_pos;
    T * curlz_pos = curlz_start_pos;
    const int * vx_curr_plane = buffer_set[0]->currSlice_data_pos;
    const int * vy_curr_plane = buffer_set[1]->currSlice_data_pos;
    const int * vz_curr_plane = buffer_set[2]->currSlice_data_pos;
    for(int i=0; i<size_x; i++){
        // prev & next planes
        const int* vy_plane_prev = vy_curr_plane - off_0;
        const int* vy_plane_next = vy_curr_plane + off_0;
        const int* vz_plane_prev = vz_curr_plane - off_0;
        const int* vz_plane_next = vz_curr_plane + off_0;
        // current planes
        const int* vx_plane_base = vx_curr_plane;
        const int* vy_plane_base = vy_curr_plane;
        const int* vz_plane_base = vz_curr_plane;
        for(size_t j=0; j<size.dim2; j++){
            size_t row_offset = j * off_1;
            // vx-dy
            const int* vx_row_prev = vx_plane_base - off_1;
            const int* vx_row_next = vx_plane_base + off_1;
            // vx-dz
            const int* vx_row = vx_plane_base;

            // vy-dx
            const int* vy_row_prev = vy_plane_prev + row_offset;
            const int* vy_row_next = vy_plane_next + row_offset;
            // vy-dz
            const int* vy_row = vy_plane_base;

            // vz-dx
            const int* vz_row_prev = vz_plane_prev + row_offset;
            const int* vz_row_next = vz_plane_next + row_offset;
            // vz-dy
            const int* vz_row_prev_2 = vz_plane_base - off_1;
            const int* vz_row_next_2 = vz_plane_base + off_1;

            const int* vz_row = vz_plane_base;
            for(size_t k=0; k<size.dim3; k++){
                curlx_pos[index] = ((vz_row_next_2[k] - vz_row_prev_2[k]) - (vy_row[k+1] - vy_row[k-1])) * errorBound;
                curly_pos[index] = ((vx_row[k+1] - vx_row[k-1]) - (vz_row_next[k] - vz_row_prev[k])) * errorBound;
                curlz_pos[index] = ((vy_row_next[k] - vy_row_prev[k]) - (vx_row_next[k] - vx_row_prev[k])) * errorBound;
                index++;
            }
            vx_plane_base += off_1;
            vy_plane_base += off_1;
            vz_plane_base += off_1;
        }
        vx_curr_plane += off_0;
        vy_curr_plane += off_0;
        vz_curr_plane += off_0;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void curlProcessBlocksPrePred(
    DSize_3d &size,
    std::array<CmpBufferSet *, 3>& cmpkit_set,
    std::array<AppBufferSet_3d *, 3>& buffer_set,
    std::array<unsigned char *, 3>& encode_pos,
    T *curlx_pos, T *curly_pos, T *curlz_pos,
    double errorBound
){
    int i;
    size_t BlockSliceSize = size.Bsize * size.dim2 * size.dim3;
    for(i=0; i<3; i++) buffer_set[i]->reset();
    size_t off_0 = buffer_set[0]->offset_0;
    size_t off_1 = buffer_set[0]->offset_1;
    int * tempBlockSlice = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockSliceSize;
        if(x == 0){
            for(i=0; i<3; i++){
                recoverBlockSlice2PrePred(x, size, encode_pos[i], buffer_set[i]->currSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                memcpy(buffer_set[i]->nextSlice_data_pos-off_0-off_1-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
            }
            curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, true, false);
        }else{
            for(i=0; i<3; i++){
                rotate_buffer(buffer_set[i]->currSlice_data_pos, buffer_set[i]->prevSlice_data_pos, buffer_set[i]->nextSlice_data_pos, tempBlockSlice);
            }
            if(x == size.block_dim1 - 1){
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, true);
            }else{
                for(i=0; i<3; i++){
                    memcpy(buffer_set[i]->nextSlice_data_pos-off_0-off_1-1, buffer_set[i]->decmp_buffer, off_0*sizeof(int));
                    recoverBlockSlice2PrePred(x+1, size, encode_pos[i], buffer_set[i]->nextSlice_data_pos, buffer_set[i], cmpkit_set[i]);
                }
                curlProcessBlockSlicePrePred(x, size, buffer_set, curlx_pos+offset, curly_pos+offset, curlz_pos+offset, errorBound, off_0, off_1, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_curl(
    std::array<unsigned char *, 3> cmpData,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *curlx_result, T *curly_result, T *curlz_result,
    decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    std::array<int *, 3> Buffer_3d = {nullptr, nullptr, nullptr};
    std::array<unsigned int *, 3> absPredError = {nullptr, nullptr, nullptr};
    std::array<T *, 3> decData = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> signFlag = {nullptr, nullptr, nullptr};
    std::array<AppBufferSet_3d *, 3> buffer_set = {nullptr, nullptr, nullptr};
    std::array<CmpBufferSet *, 3> cmpkit_set = {nullptr, nullptr, nullptr};
    std::array<unsigned char *, 3> encode_pos = {nullptr, nullptr, nullptr};
    for(int i=0; i<3; i++){
        Buffer_3d[i] = (int *)malloc(buffer_size * 4 * sizeof(int));
        absPredError[i] = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
        decData[i] = (T *)malloc(size.nbEle * sizeof(T));
        signFlag[i] = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
        buffer_set[i] = new AppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d[i]);
        cmpkit_set[i] = new CmpBufferSet(cmpData[i], absPredError[i], signFlag[i], nullptr);
        encode_pos[i] = cmpData[i] + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    }
    T * curlx_pos = curlx_result;
    T * curly_pos = curly_result;
    T * curlz_pos = curlz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            printf("recover_time = 0.0001\n");
            printf("process_time = 0.0001\n");
            break;
        }
        case decmpState::prePred:{
            curlProcessBlocksPrePred(size, cmpkit_set, buffer_set, encode_pos, curlx_pos, curly_pos, curlz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            for(int i=0; i<3; i++){
                SZp_decompress(decData[i], cmpData[i], dim1, dim2, dim3, blockSideLength, errorBound);
            }
            clock_gettime(CLOCK_REALTIME, &end2);
            rec_time += get_elapsed_time(start2, end2);
            printf("recover_time = %.6f\n", rec_time);
            clock_gettime(CLOCK_REALTIME, &start2);
            compute_curl_3d(dim1, dim2, dim3, decData[0], decData[1], decData[2], curlx_pos, curly_pos, curlz_pos);
            clock_gettime(CLOCK_REALTIME, &end2);
            op_time += get_elapsed_time(start2, end2);
            printf("process_time = %.6f\n", op_time);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    for(int i=0; i<3; i++){
        delete buffer_set[i];
        delete cmpkit_set[i];
        free(Buffer_3d[i]);
        free(absPredError[i]);
        free(signFlag[i]);
        free(decData[i]);
    }
}

#endif