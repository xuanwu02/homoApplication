#ifndef _SZP_LORENZO_PREDICTOR_3D_HPP
#define _SZP_LORENZO_PREDICTOR_3D_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "typemanager.hpp"
#include "SZp_app_utils.hpp"
#include "utils.hpp"
#include "settings.hpp"

template <class T>
void SZp_compress_3dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + buffer_dim1_offset + 1;
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
                            quant_diff = predict_lorenzo_3d(curr_data_pos++, curr_buffer_pos++, errorBound, buffer_dim0_offset, buffer_dim1_offset);
                            (*sign_pos++) = (quant_diff < 0);
                            unsigned int abs_diff = abs(quant_diff);
                            (*abs_diff_pos++) = abs_diff;
                            max_quant_diff = max_quant_diff > abs_diff ? max_quant_diff : abs_diff;
                        }
                        block_buffer_pos += buffer_dim1_offset;
                        curr_data_pos += size.dim1_offset - size_z;
                    }
                    block_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                    curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
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
            buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.dim1_offset;
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
void SZp_decompress_3dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * pred_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(pred_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = pred_buffer + buffer_dim0_offset + buffer_dim1_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            T * z_data_pos = y_data_pos;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
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
                T * curr_data_pos = z_data_pos;
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        memcpy(block_buffer_pos, signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            recover_lorenzo_3d(curr_data_pos, curr_buffer_pos, errorBound, buffer_dim0_offset, buffer_dim1_offset);
                            curr_data_pos++;
                            curr_buffer_pos++;
                        }
                        block_buffer_pos += buffer_dim1_offset;
                        curr_data_pos += size.dim1_offset - size_z;
                    }
                    block_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                    curr_data_pos += size.dim0_offset - size_y * size.dim1_offset;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size_z;
            }
            buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        memcpy(pred_buffer, pred_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
        x_data_pos += size.Bsize * size.dim0_offset;
    }
    free(pred_buffer);
    free(signPredError);
    free(signFlag);
}

double SZp_mean_3dLorenzo_recover2PostPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * diff_pos = signPredError;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (size.dim3 - (index_z + k)) * (*diff_pos++);
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
    free(signPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

double SZp_mean_3dLorenzo_recover2PrePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + buffer_dim1_offset + 1;
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                }else{
                    memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
                }
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        memcpy(block_buffer_pos, signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            recover_lorenzo_3d(quant_sum, curr_buffer_pos++, buffer_dim0_offset, buffer_dim1_offset);
                        }
                        block_buffer_pos += buffer_dim1_offset;
                    }
                    block_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
    }
    free(signPredError);
    free(signFlag);
    double mean = quant_sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean_3dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress_3dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean_3dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean_3dLorenzo_recover2PostPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            mean = SZp_mean_3dLorenzo_recover2PrePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean_3dLorenzo_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZp_variance_3dLorenzo_postPredMean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0, squared_quant_sum = 0;
    int index_x = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int index_y = 0;
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + buffer_dim1_offset + 1;
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    int * diff_pos = signPredError;
                    for(int i=0; i<size_x; i++){
                        for(int j=0; j<size_y; j++){
                            for(int k=0; k<size_z; k++){
                                quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (size.dim3 - (index_z + k)) * (*diff_pos++);
                            }
                        }
                    }
                }else{
                    memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
                }
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        memcpy(block_buffer_pos, signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            int curr_quant = recover_lorenzo_3d_verb(curr_buffer_pos++, buffer_dim0_offset, buffer_dim1_offset);
                            squared_quant_sum += curr_quant * curr_quant;
                        }
                        block_buffer_pos += buffer_dim1_offset;
                    }
                    block_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
                buffer_start_pos += size.Bsize;
                index_z += size.Bsize;
            }
            buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
            index_y += size.Bsize;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
        index_x += size.Bsize;
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1) * (2 * errorBound) * (2 * errorBound);
    return var;
}

double SZp_variance_3dLorenzo_prePredMean(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    size_t buffer_dim1_offset = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*(size.dim3+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0, squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + buffer_dim0_offset + buffer_dim1_offset + 1;
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
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signPredError, fixed_rate);
                    cmpData_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                }else{
                    memset(signPredError, 0, size.max_num_block_elements*sizeof(int));
                }
                int * block_buffer_pos = buffer_start_pos;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        memcpy(block_buffer_pos, signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                        int * curr_buffer_pos = block_buffer_pos;
                        for(int k=0; k<size_z; k++){
                            int curr_quant = recover_lorenzo_3d_verb(curr_buffer_pos++, buffer_dim0_offset, buffer_dim1_offset);
                            quant_sum += curr_quant;
                            squared_quant_sum += curr_quant * curr_quant;
                        }
                        block_buffer_pos += buffer_dim1_offset;
                    }
                    block_buffer_pos += buffer_dim0_offset - size_y * buffer_dim1_offset;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1) * (2 * errorBound) * (2 * errorBound);
    return var;
}

template <class T>
double SZp_variance_3dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress_3dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZp_variance_3dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZp_variance_3dLorenzo_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZp_variance_3dLorenzo_prePredMean(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            var = SZp_variance_3dLorenzo_postPredMean(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockPlane2PostPred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d *buffer_set, SZpCmpBufferSet *cmpkit_set
){
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
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        memcpy(curr_buffer_pos, cmpkit_set->signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                        curr_buffer_pos += buffer_set->buffer_dim1_offset;
                    }
                    curr_buffer_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
                }
            }else{
                for(int j=0; j<size_y; j++){
                    for(int k=0; k<size_z; k++){
                        memset(curr_buffer_pos, 0, size_z*sizeof(int));
                        curr_buffer_pos += buffer_set->buffer_dim1_offset;
                    }
                    curr_buffer_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
                }
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
}

template <class T>
inline void dxdydzProcessBlockPlanePostPred(
    size_t x, DSize_3d size, derivIntBuffer_3d *deriv_buffer,
    SZpAppBufferSet_3d *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopPlane, bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    size_t dx_buffer_dim0_offset = size.dim3 + 1;
    size_t dy_buffer_dim0_offset = size.dim3 + 1;
    size_t dz_buffer_dim0_offset = size.dim2 + 1;
    for(int i=0; i<size_x; i++){
        size_t global_x_offset = x * size.Bsize + i;
        T * dx_pos = dx_start_pos + i * size.dim0_offset;
        T * dy_pos = dy_start_pos + i * size.dim0_offset;
        T * dz_pos = dz_start_pos + i * size.dim0_offset;
        const int * dx_level_0_pos = isTopPlane && (i == 0) ? buffer_set->currPlane_data_pos + buffer_set->buffer_dim0_offset * (i + 1) : buffer_set->currPlane_data_pos + buffer_set->buffer_dim0_offset * i;
        const int * dx_level_1_pos = i < size_x - 1 ? buffer_set->currPlane_data_pos + buffer_set->buffer_dim0_offset * (i + 1)
                                    : isBottomPlane ? buffer_set->currPlane_data_pos + buffer_set->buffer_dim0_offset * i
                                    : buffer_set->nextPlane_data_pos;
        const int * curr_x_plane = buffer_set->currPlane_data_pos + buffer_set->buffer_dim0_offset * i;
        for(size_t j=0; j<size.dim2; j++){
            int * dx_int_buffer_pos = deriv_buffer->dx_buffer + (j + 1) * dx_buffer_dim0_offset + 1; 
            const int * dy_level_0_pos = j == 0 ? curr_x_plane + (j + 1) * buffer_set->buffer_dim1_offset : curr_x_plane + j * buffer_set->buffer_dim1_offset;
            const int * dy_level_1_pos = j == size.dim2 - 1 ? curr_x_plane + j * buffer_set->buffer_dim1_offset : curr_x_plane + (j + 1) * buffer_set->buffer_dim1_offset;
            int * dy_int_buffer_pos = deriv_buffer->dy_buffer[j] + dy_buffer_dim0_offset * (global_x_offset + 1) + 1;
            const int * curr_y_row = curr_x_plane + j * buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                const int * dz_level_0_pos = k == 0 ? curr_y_row + k + 1 : curr_y_row + k;
                const int * dz_level_1_pos = k == size.dim3 - 1 ? curr_y_row + k : curr_y_row + k + 1;
                int * dz_int_buffer_pos = deriv_buffer->dz_buffer[k] + dz_buffer_dim0_offset * (global_x_offset + 1) + j + 1;
                deriv_lorenzo_2d(dx_level_0_pos++, dx_level_1_pos++, dx_int_buffer_pos++, dx_pos++, dx_buffer_dim0_offset, errorBound);
                deriv_lorenzo_2d(dy_level_0_pos++, dy_level_1_pos++, dy_int_buffer_pos++, dy_pos++, dy_buffer_dim0_offset, errorBound);
                deriv_lorenzo_2d(dz_level_0_pos, dz_level_1_pos, dz_int_buffer_pos, dz_pos++, dz_buffer_dim0_offset, errorBound);
            }
        }
    }
}

template <class T>
inline void dxdydzProcessBlocksPostPred(
    DSize_3d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_3d *buffer_set,
    derivIntBuffer_3d *deriv_buffer,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockPlane = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PostPred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set);
            recoverBlockPlane2PostPred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
            dxdydzProcessBlockPlanePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, true, false);
        }else{
            std::swap(buffer_set->currPlane_data_pos, buffer_set->nextPlane_data_pos);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockPlanePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, true);
            }else{
                recoverBlockPlane2PostPred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
                dxdydzProcessBlockPlanePostPred(x, size, deriv_buffer, buffer_set, errorBound, dx_pos+offset, dy_pos+offset, dz_pos+offset, false, false);
            }
        }
    }
}

inline void recoverBlockPlane2PrePred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d *buffer_set, SZpCmpBufferSet *cmpkit_set
){
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * curr_buffer_pos = buffer_start_pos;
            int * quant_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
            }else{
                memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
            }
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    memcpy(curr_buffer_pos, cmpkit_set->signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                    for(int k=0; k<size_z; k++){
                        int * quant_ptr = quant_pos + k;
                        recover_lorenzo_3d(quant_ptr, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                    }
                    curr_buffer_pos += buffer_set->buffer_dim1_offset;
                    quant_pos += buffer_set->buffer_dim1_offset;
                }
                curr_buffer_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
                quant_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
    memcpy(buffer_set->decmp_buffer, buffer_data_pos+(size.Bsize-1)*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, buffer_set->buffer_dim0_offset*sizeof(int));
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred(
    size_t x, DSize_3d& size, SZpAppBufferSet_3d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopPlane, bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bsize*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * x_dx_pos = dx_start_pos + i * size.dim0_offset;
        T * x_dy_pos = dy_start_pos + i * size.dim0_offset;
        T * x_dz_pos = dz_start_pos + i * size.dim0_offset;
        const int * prev_plane = curr_plane - buffer_set->buffer_dim0_offset;
        const int * next_plane = curr_plane + buffer_set->buffer_dim0_offset;
        const int * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_dx_pos = x_dx_pos + j * size.dim1_offset;
            T * y_dy_pos = x_dy_pos + j * size.dim1_offset;
            T * y_dz_pos = x_dz_pos + j * size.dim1_offset;
            const int * prev_row = curr_row - buffer_set->buffer_dim1_offset;
            const int * next_row = curr_row + buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                // size_t buffer_index_1d = k + 1;
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_dx_pos[index_1d] = (next_plane[buffer_index_2d] - prev_plane[buffer_index_2d]) * errorBound;
                y_dy_pos[index_1d] = (next_row[index_1d] - prev_row[index_1d]) * errorBound;
                y_dz_pos[index_1d] = (curr_row[index_1d + 1] - curr_row[index_1d - 1]) * errorBound;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
    // // legacy
    // size_t buffer_dim0_offset = (size.dim2 + 1) * (size.dim3 + 1);
    // size_t buffer_dim1_offset = size.dim3 + 1;
    // const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_dim0_offset - buffer_dim1_offset - 1;
    // const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_dim1_offset - 1;
    // const int * curr_plane = buffer_set->currPlane_data_pos - buffer_dim1_offset - 1;
    // for(int i=0; i<size_x; i++){
    //     T * dx_pos = dx_start_pos + i * size.dim0_offset;
    //     T * dy_pos = dy_start_pos + i * size.dim0_offset;
    //     T * dz_pos = dz_start_pos + i * size.dim0_offset;
    //     const int * prev_plane = i > 0 ? curr_plane - buffer_dim0_offset
    //                            : isTopPlane ? curr_plane : prevBlockPlaneBottom_pos;
    //     const int * next_plane = i < size_x - 1 ? curr_plane + buffer_dim0_offset
    //                            : isBottomPlane ? curr_plane : nextBlockPlaneTop_pos;
    //     const int * curr_row = curr_plane + buffer_dim1_offset + 1;
    //     int coeff_dx = (isTopPlane && i == 0) || (isBottomPlane && i == size_x - 1) ? 2 : 1;
    //     for(size_t j=0; j<size.dim2; j++){
    //         const int * prev_row = j == 0 ? curr_row : curr_row - buffer_dim1_offset;
    //         const int * next_row = j == size.dim2 - 1 ? curr_row : curr_row + buffer_dim1_offset;
    //         int coeff_dy = (j == 0) || (j == size.dim2 - 1) ? 2 : 1;
    //         for(size_t k=0; k<size.dim3; k++){
    //             size_t buffer_index = (j + 1) * buffer_dim1_offset + k + 1;
    //             size_t res_index = j * size.dim1_offset + k;
    //             size_t prev_k = k == 0 ? k : k - 1;
    //             size_t next_k = k == size.dim3 - 1 ? k : k + 1;
    //             int coeff_dz = (k == 0) || (k == size.dim3 - 1) ? 2 : 1;
    //             dx_pos[res_index] = (next_plane[buffer_index] - prev_plane[buffer_index]) * coeff_dx * errorBound;
    //             dy_pos[res_index] = (next_row[k] - prev_row[k]) * coeff_dy * errorBound;
    //             dz_pos[res_index] = (curr_row[next_k] - curr_row[prev_k]) * coeff_dz * errorBound;
    //         }
    //         curr_row += buffer_dim1_offset;
    //     }
    //     curr_plane += buffer_dim0_offset;
    // }
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d &size,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_3d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockPlane = nullptr;
    for(size_t x=0; x<size.block_dim1; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set);
            memcpy(buffer_set->nextPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset*sizeof(int));
            recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
            dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == size.block_dim1 - 1){
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, true);
            }else{
                memcpy(buffer_set->nextPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, buffer_set->decmp_buffer, buffer_set->buffer_dim0_offset*sizeof(int));
                recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, false);
            }
        }
    }
}

template <class T>
void SZp_dxdydz_3dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    size_t dim3, int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, T *dz_result, decmpState state
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * Buffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    SZpAppBufferSet_3d * buffer_set = new SZpAppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, Buffer_2d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
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
            SZp_decompress_3dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            compute_dxdydz(dim1, dim2, dim3, decData, dx_pos, dy_pos, dz_pos);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_3d);
    free(Buffer_2d);
    free(signPredError);
    free(signFlag);
    free(decData);
}

/**
 * Use 2dlorenzo in 3d data
 */
template <class T>
void SZp_compress_2dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    DSize_3d2d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = size.dim3 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim3+1)*sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    const T * x_data_pos = oriData;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        memset(quant_buffer, 0, (size.Bsize+1)*(size.dim3+1)*sizeof(int));
        for(size_t y=0; y<size.block_dim2; y++){
            const T * z_data_pos = y_data_pos;
            int * buffer_start_pos = quant_buffer + buffer_dim0_offset + 1;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_y * size_z;
                unsigned int * abs_err_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                int max_err = 0;
                int * block_buffer_pos = buffer_start_pos;
                const T * curr_data_pos = z_data_pos;
                for(int j=0; j<size_y; j++){
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int k=0; k<size_z; k++){
                        int err = predict_lorenzo_2d(curr_data_pos++, curr_buffer_pos++, buffer_dim0_offset, errorBound);
                        (*sign_pos++) = (err < 0);
                        unsigned int abs_err = abs(err);
                        (*abs_err_pos++) = abs_err;
                        max_err = max_err > abs_err ? max_err : abs_err;
                    }
                    block_buffer_pos += buffer_dim0_offset;
                    curr_data_pos += size.dim1_offset - size_z;
                }
                int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
                cmpData[block_ind++] = (unsigned char)fixed_rate;
                if(fixed_rate){
                    unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                    cmpData_pos += signbyteLength;
                    unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, cmpData_pos, fixed_rate);
                    cmpData_pos += savedbitsbyteLength;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size.Bsize;
            }
            memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.dim0_offset;
    }
    cmpSize = cmpData_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_decompress_2dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d2d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim0_offset = size.dim3 + 1;
    int * err_buffer = (int *)malloc((size.Bsize+1)*(size.dim3+1)*sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    T * x_data_pos = decData;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        memset(err_buffer, 0, (size.Bsize+1)*(size.dim3+1)*sizeof(int));
        for(size_t y=0; y<size.block_dim2; y++){
            T * z_data_pos = y_data_pos;
            int * buffer_start_pos = err_buffer + buffer_dim0_offset + 1;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                int * block_buffer_pos = buffer_start_pos;
                T * curr_data_pos = z_data_pos;
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
                for(int j=0; j<size_y; j++){
                    memcpy(block_buffer_pos, signPredError+j*size_z, size_z*sizeof(int));
                    int * curr_buffer_pos = block_buffer_pos;
                    for(int k=0; k<size_z; k++){
                        recover_lorenzo_2d(curr_data_pos++, curr_buffer_pos++, buffer_dim0_offset, errorBound);
                    }
                    block_buffer_pos += buffer_dim0_offset;
                    curr_data_pos += size.dim1_offset - size_z;
                }
                buffer_start_pos += size.Bsize;
                z_data_pos += size.Bsize;
            }
            memcpy(err_buffer, err_buffer+size.Bsize*buffer_dim0_offset, buffer_dim0_offset*sizeof(int));
            y_data_pos += size.Bsize * size.dim1_offset;
        }
        x_data_pos += size.dim0_offset;
    }
    free(err_buffer);
    free(signPredError);
    free(signFlag);
}

inline void recoverBlockPlane2PostPred(
    size_t x, DSize_3d2d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d *buffer_set, SZpCmpBufferSet *cmpkit_set, double errorBound
){
    int block_ind = x * size.Bwidth * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    for(int i=0; i<size_x; i++){
        int * buffer_start_pos = buffer_data_pos + i * buffer_set->buffer_dim0_offset;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_y * size_z;
                int * block_buffer_pos = buffer_start_pos;
                int * err_pos = buffer_start_pos;
                int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                }else{
                    memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
                }
                for(int j=0; j<size_y; j++){
                    memcpy(block_buffer_pos, cmpkit_set->signPredError+j*size_z, size_z*sizeof(int));
                    for(int k=0; k<size_z; k++){
                        recover_lorenzo_2d(err_pos+k, buffer_set->buffer_dim1_offset);
                    }
                    block_buffer_pos += buffer_set->buffer_dim1_offset;
                    err_pos += buffer_set->buffer_dim1_offset;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
    }
}

inline void recoverBlockPlane2PrePred(
    size_t x, DSize_3d2d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d *buffer_set, SZpCmpBufferSet *cmpkit_set, double errorBound
){
    int block_ind = x * size.Bwidth * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    for(int i=0; i<size_x; i++){
        int * buffer_start_pos = buffer_data_pos + i * buffer_set->buffer_dim0_offset;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_y * size_z;
                int * block_buffer_pos = buffer_start_pos;
                int * err_pos = buffer_start_pos;
                int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                }else{
                    memset(cmpkit_set->signPredError, 0, block_size*sizeof(int));
                }
                for(int j=0; j<size_y; j++){
                    memcpy(block_buffer_pos, cmpkit_set->signPredError+j*size_z, size_z*sizeof(int));
                    for(int k=0; k<size_z; k++){
                        recover_lorenzo_2d(err_pos+k, buffer_set->buffer_dim1_offset);
                    }
                    block_buffer_pos += buffer_set->buffer_dim1_offset;
                    err_pos += buffer_set->buffer_dim1_offset;
                }
                buffer_start_pos += size.Bsize;
            }
            buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
    }
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred(
    size_t x, DSize_3d2d& size, SZpAppBufferSet_3d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopPlane, bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bsize - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * dx_pos = dx_start_pos + i * size.dim0_offset;
        T * dy_pos = dy_start_pos + i * size.dim0_offset;
        T * dz_pos = dz_start_pos + i * size.dim0_offset;
        const int * prev_plane = isTopPlane && i == 0 ? curr_plane
                                 : i == 0 ? prevBlockPlaneBottom_pos
                                 : curr_plane - buffer_set->buffer_dim0_offset;
        const int * next_plane = isBottomPlane && i == size_x - 1 ? curr_plane
                                 : i == size_x - 1 ? nextBlockPlaneTop_pos
                                 : curr_plane + buffer_set->buffer_dim0_offset;
        const int * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        int coeff_dx = (isTopPlane && i == 0) || (isBottomPlane && i == size_x - 1) ? 2 : 1;
        for(size_t j=0; j<size.dim2; j++){
            const int * prev_row = j == 0 ? curr_row : curr_row - buffer_set->buffer_dim1_offset;
            const int * next_row = j == size.dim2 - 1 ? curr_row : curr_row + buffer_set->buffer_dim1_offset;
            int coeff_dy = (j == 0) || (j == size.dim2 - 1) ? 2 : 1;
            for(size_t k=0; k<size.dim3; k++){
                size_t buffer_index = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                size_t res_index = j * size.dim1_offset + k;
                size_t prev_k = k == 0 ? k : k - 1;
                size_t next_k = k == size.dim3 - 1 ? k : k + 1;
                int coeff_dz = (k == 0) || (k == size.dim3 - 1) ? 2 : 1;
                dx_pos[res_index] = (next_plane[buffer_index] - prev_plane[buffer_index]) * coeff_dx * errorBound;
                dy_pos[res_index] = (next_row[k] - prev_row[k]) * coeff_dy * errorBound;
                dz_pos[res_index] = (curr_row[next_k] - curr_row[prev_k]) * coeff_dz * errorBound;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d2d &size,
    size_t numBlockPlane,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_3d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bsize * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockPlane = nullptr;
    for(size_t x=0; x<numBlockPlane; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set, errorBound);
            recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set, errorBound);
            dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == numBlockPlane - 1){
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, true);
            }else{
                recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set, errorBound);
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, false);
            }
        }
    }
}

template <class T>
void SZp_dxdydz_2dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    size_t dim3, int blockSideLength, double errorBound,
    T *dx_result, T *dy_result, T *dz_result, decmpState state
){
    DSize_3d2d size(dim1, dim2, dim3, blockSideLength);
    size_t numBlockPlane = (size.dim1 - 1) / size.Bsize + 1;
    size_t buffer_dim1 = size.Bsize + 1;
    size_t buffer_dim2 = size.dim2 + 1;
    size_t buffer_dim3 = size.dim3 + 1;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * Buffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    SZpAppBufferSet_3d * buffer_set = new SZpAppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, Buffer_2d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            elapsed_time = -1;
            exit(0);
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, numBlockPlane, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            SZp_decompress_2dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            compute_dxdydz(dim1, dim2, dim3, decData, dx_pos, dy_pos, dz_pos);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_3d);
    free(Buffer_2d);
    free(signPredError);
    free(signFlag);
    free(decData);
}

// gray-scott 3d lorenzo
template <class bufferType>
inline void recoverBlockPlane2PrePred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    bufferType *buffer_set, SZpCmpBufferSet *cmpkit_set, int current, int iter
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->set_decmp_buffer_border(buffer_data_pos, size_x);
    int block_ind = x * size.block_dim2 * size.block_dim3;
    int * buffer_start_pos = buffer_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            int * curr_buffer_pos = buffer_start_pos;
            int * quant_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->cmpData[current][block_ind++];
            gs_decode_prepred(block_size, fixed_rate, encode_pos, cmpkit_set);
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    memcpy(curr_buffer_pos, cmpkit_set->signPredError+i*size_y*size_z+j*size_z, size_z*sizeof(int));
                    for(int k=0; k<size_z; k++){
                        recover_lorenzo_3d(quant_pos+k, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                    }
                    curr_buffer_pos += buffer_set->buffer_dim1_offset;
                    quant_pos += buffer_set->buffer_dim1_offset;
                }
                curr_buffer_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
                quant_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
            }
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
    buffer_set->save_decmp_buffer_bottom(buffer_data_pos, size.Bsize);
}

inline void grayscottProcessBlockPlanePrePred(
    size_t x, DSize_3d& size,
    GrayScott *gs,
    gsAppBufferSet *uAppBuffer,
    gsAppBufferSet *vAppBuffer,
    double errorBound, int iter,
    bool isTopPlane,
    bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    uAppBuffer->set_process_buffer(x, isTopPlane, isBottomPlane, size_x, gs->UBorderVal);
    vAppBuffer->set_process_buffer(x, isTopPlane, isBottomPlane, size_x, gs->VBorderVal);
    const int * u_buffer_start_pos = uAppBuffer->currPlane_data_pos;
    const int * v_buffer_start_pos = vAppBuffer->currPlane_data_pos;
    int * u_update_start_pos = uAppBuffer->updatePlane_data_pos;
    int * v_update_start_pos = vAppBuffer->updatePlane_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            const int * u_block_buffer_pos = u_buffer_start_pos, * v_block_buffer_pos = v_buffer_start_pos;
            int * u_update_pos = u_update_start_pos, * v_update_pos = v_update_start_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    const int * u_curr_buffer_pos = u_block_buffer_pos;
                    const int * v_curr_buffer_pos = v_block_buffer_pos;
                    int * u_next = u_update_pos;
                    int * v_next = v_update_pos;
                    for(int k=0; k<size_z; k++){
                        int64_t tu = *u_curr_buffer_pos;
                        int64_t tv = *v_curr_buffer_pos;
                        int64_t u_lap = laplacian(u_curr_buffer_pos++, uAppBuffer->buffer_dim0_offset, uAppBuffer->buffer_dim1_offset);
                        int64_t v_lap = laplacian(v_curr_buffer_pos++, vAppBuffer->buffer_dim0_offset, vAppBuffer->buffer_dim1_offset);
                        int u_ = tu + gs->Dudt * u_lap - tu * tv * tv * gs->ebdt + gs->Fdt_int - gs->Fdt_fl * tu;
                        int v_ = tv + gs->Dvdt * v_lap + tu * tv * tv * gs->ebdt - gs->Fkdt * tv;
                        *u_next++ = u_;
                        *v_next++ = v_;
                    }
                    u_block_buffer_pos += uAppBuffer->buffer_dim1_offset;
                    v_block_buffer_pos += vAppBuffer->buffer_dim1_offset;
                    u_update_pos += uAppBuffer->buffer_dim1_offset;
                    v_update_pos += vAppBuffer->buffer_dim1_offset;
                }
                u_block_buffer_pos += uAppBuffer->buffer_dim0_offset - size_y * uAppBuffer->buffer_dim1_offset;
                v_block_buffer_pos += vAppBuffer->buffer_dim0_offset - size_y * vAppBuffer->buffer_dim1_offset;
                u_update_pos += uAppBuffer->buffer_dim0_offset - size_y * uAppBuffer->buffer_dim1_offset;
                v_update_pos += vAppBuffer->buffer_dim0_offset - size_y * vAppBuffer->buffer_dim1_offset;
            }
            u_buffer_start_pos += size.Bsize;
            v_buffer_start_pos += size.Bsize;
            u_update_start_pos += size.Bsize;
            v_update_start_pos += size.Bsize;
        }
        u_buffer_start_pos += size.Bsize * uAppBuffer->buffer_dim1_offset - size.Bsize * size.block_dim3;
        v_buffer_start_pos += size.Bsize * vAppBuffer->buffer_dim1_offset - size.Bsize * size.block_dim3;
        u_update_start_pos += size.Bsize * vAppBuffer->buffer_dim1_offset - size.Bsize * size.block_dim3;
        v_update_start_pos += size.Bsize * vAppBuffer->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
}

template <class bufferType>
inline void compressBlockPlaneFromPrePred(
    size_t x, DSize_3d& size,
    bufferType *buffer_set,
    SZpCmpBufferSet *cmpkit_set,
    int next, int iter,
    bool isTopRow
){
    buffer_set->set_cmp_buffer_top(isTopRow);
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int block_ind = x * size.block_dim2 * size.block_dim3;
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = cmpData_pos;
    const int * update_start_pos = buffer_set->updatePlane_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            int block_size = size_x * size_y * size_z;
            const int * block_quant_pos = update_start_pos;
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    const int * curr_quant_pos = block_quant_pos;
                    for(int k=0; k<size_z; k++){
                        int err = predict_lorenzo_3d(curr_quant_pos++, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                        *sign_pos++ = (err < 0);
                        abs_err = abs(err);
                        *abs_err_pos++ = abs_err;
                        max_err = max_err > abs_err ? max_err : abs_err;
                    }
                    block_quant_pos += buffer_set->buffer_dim1_offset;
                }
                block_quant_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
            }
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, cmpData_pos);
                cmpData_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, cmpData_pos, fixed_rate);
                cmpData_pos += savedbitsbyteLength;
            }
            update_start_pos += size.Bsize;
        }
        update_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
    buffer_set->save_cmp_buffer_buttom(size_x);
    size_t increment = cmpData_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void grayscottUpdatePrePred(
    DSize_3d& size,
    GrayScott *gs,
    gsAppBufferSet *uAppBuffer,
    gsAppBufferSet *vAppBuffer,
    SZpCmpBufferSet *uCmpkit,
    SZpCmpBufferSet *vCmpkit,
    double errorBound,
    int current, int next,
    int iter
){
    unsigned char * u_cmp_data = uCmpkit->cmpData[current];
    unsigned char * v_cmp_data = vCmpkit->cmpData[current];
    unsigned char * u_encode_pos = u_cmp_data + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * v_encode_pos = v_cmp_data + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * temp = nullptr;
    uCmpkit->reset(), uAppBuffer->reset();
    vCmpkit->reset(), vAppBuffer->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        if(x == 0){
            unsigned char * tmp = uCmpkit->cmpData[current];
            recoverBlockPlane2PrePred(x, size, u_encode_pos, uAppBuffer->currPlane_data_pos, uAppBuffer, uCmpkit, current, iter);
            uAppBuffer->set_next_decmp_buffer_top(uAppBuffer->nextPlane_data_pos);
            recoverBlockPlane2PrePred(x+1, size, u_encode_pos, uAppBuffer->nextPlane_data_pos, uAppBuffer, uCmpkit, current, iter);
            recoverBlockPlane2PrePred(x, size, v_encode_pos, vAppBuffer->currPlane_data_pos, vAppBuffer, vCmpkit, current, iter);
            vAppBuffer->set_next_decmp_buffer_top(vAppBuffer->nextPlane_data_pos);
            recoverBlockPlane2PrePred(x+1, size, v_encode_pos, vAppBuffer->nextPlane_data_pos, vAppBuffer, vCmpkit, current, iter);
            grayscottProcessBlockPlanePrePred(x, size, gs, uAppBuffer, vAppBuffer, errorBound, iter, true, false);
            compressBlockPlaneFromPrePred(x, size, uAppBuffer, uCmpkit, next, iter, true);
            compressBlockPlaneFromPrePred(x, size, vAppBuffer, vCmpkit, next, iter, true);
        }else{
            rotate_buffer(uAppBuffer->currPlane_data_pos, uAppBuffer->prevPlane_data_pos, uAppBuffer->nextPlane_data_pos, temp);
            rotate_buffer(vAppBuffer->currPlane_data_pos, vAppBuffer->prevPlane_data_pos, vAppBuffer->nextPlane_data_pos, temp);
            if(x == size.block_dim1 - 1){
                grayscottProcessBlockPlanePrePred(x, size, gs, uAppBuffer, vAppBuffer, errorBound, iter, false, true);
                compressBlockPlaneFromPrePred(x, size, uAppBuffer, uCmpkit, next, iter, false);
                compressBlockPlaneFromPrePred(x, size, vAppBuffer, vCmpkit, next, iter, false);
            }else{
                uAppBuffer->set_next_decmp_buffer_top(uAppBuffer->nextPlane_data_pos);
                recoverBlockPlane2PrePred(x+1, size, u_encode_pos, uAppBuffer->nextPlane_data_pos, uAppBuffer, uCmpkit, current, iter);
                vAppBuffer->set_next_decmp_buffer_top(vAppBuffer->nextPlane_data_pos);
                recoverBlockPlane2PrePred(x+1, size, v_encode_pos, vAppBuffer->nextPlane_data_pos, vAppBuffer, vCmpkit, current, iter);
                grayscottProcessBlockPlanePrePred(x, size, gs, uAppBuffer, vAppBuffer, errorBound, iter, false, false);
                compressBlockPlaneFromPrePred(x, size, uAppBuffer, uCmpkit, next, iter, false);
                compressBlockPlaneFromPrePred(x, size, vAppBuffer, vCmpkit, next, iter, false);
            }
        }
    }
}

template <class T>
inline void grayscottUpdatePrePred(
    DSize_3d& size,
    GrayScott *gs,
    gsAppBufferSet *uAppBuffer,
    gsAppBufferSet *vAppBuffer,
    SZpCmpBufferSet *uCmpkit,
    SZpCmpBufferSet *vCmpkit,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t u_cmpSize, v_cmpSize;
    T * u = (T *)malloc(size.nbEle * sizeof(T));
    T * v = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        if(verb){
            if(iter >= gs_plot_offset && iter % gs_plot_gap == 0){
                SZp_decompress_3dLorenzo(u, uCmpkit->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
                SZp_decompress_3dLorenzo(v, vCmpkit->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
                std::string u_name = grayscott_data_dir + "/u.pre." + std::to_string(iter-1);
                std::string v_name = grayscott_data_dir + "/v.pre." + std::to_string(iter-1);
                writefile(u_name.c_str(), u, size.nbEle);
                writefile(v_name.c_str(), v, size.nbEle);
                u_cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + uCmpkit->cmpSize;
                printf("prepred iter %d: u_cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / u_cmpSize);
                v_cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + vCmpkit->cmpSize;
                printf("prepred iter %d: v_cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / v_cmpSize);
                fflush(stdout);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        grayscottUpdatePrePred(size, gs, uAppBuffer, vAppBuffer, uCmpkit, vCmpkit, errorBound, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZp_decompress_3dLorenzo(u, uCmpkit->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
        SZp_decompress_3dLorenzo(v, vCmpkit->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
        std::string u_name = grayscott_data_dir + "/u.pre." + std::to_string(iter);
        std::string v_name = grayscott_data_dir + "/v.pre." + std::to_string(iter);
        writefile(u_name.c_str(), u, size.nbEle);
        writefile(v_name.c_str(), v, size.nbEle);
        u_cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + uCmpkit->cmpSize;
        printf("prepred exit u_cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / u_cmpSize);
        v_cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + vCmpkit->cmpSize;
        printf("prepred exit v_cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / v_cmpSize);
    }
    printf("prepred elapsed_time = %.6f\n", elapsed_time);
    free(u);
    free(v);
}

template <class T>
inline void grayscottUpdateDOC(
    DSize_3d& size,
    size_t dim1_padded,
    size_t dim2_padded,
    size_t dim3_padded,
    size_t nbEle_padded,
    GrayScott *grayscott,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t u_cmpSize, v_cmpSize;
    T * u = (T *)malloc(nbEle_padded * sizeof(T));
    T * v = (T *)malloc(nbEle_padded * sizeof(T));
    T * u2 = (T *)malloc(nbEle_padded * sizeof(T));
    T * v2 = (T *)malloc(nbEle_padded * sizeof(T));
    unsigned char * u_compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    unsigned char * v_compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    grayscott->initData(u, v, u2, v2);
    SZp_compress_3dLorenzo(u, u_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, u_cmpSize);
    SZp_compress_3dLorenzo(v, v_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, v_cmpSize);
    T * tmp = nullptr;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        clock_gettime(CLOCK_REALTIME, &start);
        SZp_decompress_3dLorenzo(u, u_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound);
        SZp_decompress_3dLorenzo(v, v_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter >= gs_plot_offset && iter % gs_plot_gap == 0){
                std::string u_name = grayscott_data_dir + "/u.doc." + std::to_string(iter-1);
                std::string v_name = grayscott_data_dir + "/v.doc." + std::to_string(iter-1);
                writefile(u_name.c_str(), u, nbEle_padded);
                writefile(v_name.c_str(), v, nbEle_padded);
                printf("doc iter %d: u_cr = %.2f\n", iter-1, 1.0 * nbEle_padded * sizeof(T) / u_cmpSize);
                printf("doc iter %d: v_cr = %.2f\n", iter-1, 1.0 * nbEle_padded * sizeof(T) / v_cmpSize);
                fflush(stdout);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        grayscott->iterate(u, v, u2, v2, tmp);
        SZp_compress_3dLorenzo(u, u_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, u_cmpSize);
        SZp_compress_3dLorenzo(v, v_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, v_cmpSize);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZp_decompress_3dLorenzo(u, u_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound);
        SZp_decompress_3dLorenzo(v, v_compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound);
        std::string u_name = grayscott_data_dir + "/u.doc." + std::to_string(iter);
        std::string v_name = grayscott_data_dir + "/v.doc." + std::to_string(iter);
        writefile(u_name.c_str(), u, nbEle_padded);
        writefile(v_name.c_str(), v, nbEle_padded);
        printf("doc exit u_cr = %.2f\n", 1.0 * nbEle_padded * sizeof(T) / u_cmpSize);
        printf("doc exit v_cr = %.2f\n", 1.0 * nbEle_padded * sizeof(T) / v_cmpSize);
    }
    printf("doc elapsed_time = %.6f\n", elapsed_time);
    free(u_compressed);
    free(v_compressed);
    free(u);
    free(v);
    free(u2);
    free(v2);
}

template <class T>
void SZp_grayscott_3dLorenzo(
    double Du, double Dv,
    double F, double k, double dt,
    unsigned char *uCmpDataBuffer,
    unsigned char *vCmpDataBuffer,
    size_t L, int blockSideLength,
    int max_iter, double errorBound,
    decmpState state, bool verb
){
    DSize_3d size(L, L, L, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    size_t dim1_padded = size.dim1 + 2;
    size_t nbEle_padded = dim1_padded * buffer_dim2 * buffer_dim3;
    int * uBuffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    int * uBuffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * 2 * sizeof(int));
    int * vBuffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    int * vBuffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * 2 * sizeof(int));
    unsigned int * uAbsPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * uSignPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * uSignFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
    unsigned int * vAbsPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * vSignPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * vSignFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));

    unsigned char **uCmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    unsigned char **vCmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **uOffsets = (int **)malloc(2*sizeof(int *));
    int **vOffsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        uCmpData[i] = (unsigned char *)malloc(nbEle_padded * sizeof(T));
        vCmpData[i] = (unsigned char *)malloc(nbEle_padded * sizeof(T));
        uOffsets[i] = (int *)malloc(size.block_dim1 * sizeof(int));
        vOffsets[i] = (int *)malloc(size.block_dim1 * sizeof(int));
    }
    memcpy(uCmpData[0], uCmpDataBuffer, nbEle_padded * sizeof(T));
    memcpy(vCmpData[0], vCmpDataBuffer, nbEle_padded * sizeof(T));

    size_t u_prefix_length = 0, v_prefix_length = 0;
    int block_index = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        uOffsets[0][x] = u_prefix_length, vOffsets[0][x] = v_prefix_length;
        uOffsets[1][x] = 0, vOffsets[1][x] = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim2; z++){
                int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
                int size_y = ((y+1) * size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y * size.Bsize;
                int size_z = ((z+1) * size.Bsize < size.dim2) ? size.Bsize : size.dim2 - z * size.Bsize;
                int block_size = size_x * size_y * size_z;
                int cmp_block_sign_length = (block_size + 7) / 8;
                int u_fixed_rate = (int)uCmpDataBuffer[block_index];
                int v_fixed_rate = (int)vCmpDataBuffer[block_index];
                size_t uBytes = compute_encoding_byteLength(block_size, u_fixed_rate);
                size_t vBytes = compute_encoding_byteLength(block_size, v_fixed_rate);
                if(u_fixed_rate)
                    u_prefix_length += (cmp_block_sign_length + uBytes);
                if(v_fixed_rate)
                    v_prefix_length += (cmp_block_sign_length + vBytes);
                block_index++;
            }
        }
    }

    SZpCmpBufferSet * uCmpkit = new SZpCmpBufferSet(uCmpData, uOffsets, uAbsPredError, uSignPredError, uSignFlag);
    SZpCmpBufferSet * vCmpkit = new SZpCmpBufferSet(vCmpData, vOffsets, vAbsPredError, vSignPredError, vSignFlag);
    gsAppBufferSet * uAppBuffer = new gsAppBufferSet(buffer_dim1, buffer_dim2, buffer_dim3, uBuffer_3d, uBuffer_2d);
    gsAppBufferSet * vAppBuffer = new gsAppBufferSet(buffer_dim1, buffer_dim2, buffer_dim3, vBuffer_3d, vBuffer_2d);
    GrayScott * grayscott = new GrayScott(L, Du, Dv, dt, F, k, errorBound);

    switch(state){
        case decmpState::full:{
            grayscottUpdateDOC<T>(size, dim1_padded, buffer_dim2, buffer_dim3, nbEle_padded, grayscott, errorBound, max_iter, verb);
            break;
        }
        case decmpState::prePred:{
            grayscottUpdatePrePred<T>(size, grayscott, uAppBuffer, vAppBuffer, uCmpkit, vCmpkit, errorBound, max_iter, verb);
            break;
        }
        case decmpState::postPred:{
            printf("Post-prediction state does not apply to gray-scott\n");
            break;
        }
    }

    delete uCmpkit;
    delete vCmpkit;
    delete uAppBuffer;
    delete vAppBuffer;
    delete grayscott;
    for(int i=0; i<2; i++){
        free(uCmpData[i]);
        free(vCmpData[i]);
        free(uOffsets[i]);
        free(vOffsets[i]);
    }
    free(uCmpData);
    free(vCmpData);
    free(uOffsets);
    free(vOffsets);
    free(uBuffer_3d);
    free(uBuffer_2d);
    free(vBuffer_3d);
    free(vBuffer_2d);
    free(uAbsPredError);
    free(uSignPredError);
    free(uSignFlag);
    free(vAbsPredError);
    free(vSignPredError);
    free(vSignFlag);
}

template <class T>
void SZp_grayscott_3dLorenzo(
    unsigned char *uCmpDataBuffer,
    unsigned char *vCmpDataBuffer,
    gsSettings& s, decmpState state, bool verb
){
    SZp_grayscott_3dLorenzo<T>(s.Du, s.Dv, s.F, s.k, s.dt, uCmpDataBuffer, vCmpDataBuffer, s.L, s.B, s.steps, s.eb, state, verb);
}

// 3d heat diffusion
inline void heatdisProcessBlockPlanePrePred(
    size_t x, DSize_3d& size,
    ht3DBufferSet *buffer_set,
    TempInfo3D &temp_info,
    float alpha,
    double errorBound, int iter,
    bool isTopPlane,
    bool isBottomPlane
){
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    buffer_set->set_process_buffer(x, isTopPlane, isBottomPlane, size_x, 0);
    set_buffer_border_prepred(buffer_set->currPlane_data_pos, size, size_x, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset, temp_info, isTopPlane, isBottomPlane);
    const int * buffer_start_pos = buffer_set->currPlane_data_pos;
    int * update_start_pos = buffer_set->updatePlane_data_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        for(size_t z=0; z<size.block_dim3; z++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
            const int * block_buffer_pos = buffer_start_pos;
            int * update_pos = update_start_pos;
            for(int i=0; i<size_x; i++){
                for(int j=0; j<size_y; j++){
                    const int * curr_buffer_pos = block_buffer_pos;
                    int * h_next = update_pos;
                    for(int k=0; k<size_z; k++){
                        int64_t th = *curr_buffer_pos;
                        int h_ = th + alpha * laplacian(curr_buffer_pos++, buffer_set->buffer_dim0_offset, buffer_set->buffer_dim1_offset);
                        *h_next++ = h_;
                    }
                    block_buffer_pos += buffer_set->buffer_dim1_offset;
                    update_pos += buffer_set->buffer_dim1_offset;
                }
                block_buffer_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
                update_pos += buffer_set->buffer_dim0_offset - size_y * buffer_set->buffer_dim1_offset;
            }
            buffer_start_pos += size.Bsize;
            update_start_pos += size.Bsize;
        }
        buffer_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
        update_start_pos += size.Bsize * buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
    }
}

inline void heatdisUpdatePrePred(
    DSize_3d& size,
    HeatDis3D *heatdis,
    TempInfo3D& temp_info,
    ht3DBufferSet *buffer_set,
    SZpCmpBufferSet *cmpkit_set,
    double errorBound,
    int current, int next,
    int iter
){
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int * temp = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    for(size_t x=0; x<size.block_dim1; x++){
        if(x == 0){
            recoverBlockPlane2PrePred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set, current, iter);
            buffer_set->set_next_decmp_buffer_top(buffer_set->nextPlane_data_pos);
            recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set, current, iter);
            heatdisProcessBlockPlanePrePred(x, size, buffer_set, temp_info, heatdis->alpha, errorBound, iter, true, false);
            compressBlockPlaneFromPrePred(x, size, buffer_set, cmpkit_set, next, iter, true);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, temp);
            if(x == size.block_dim1 - 1){
                heatdisProcessBlockPlanePrePred(x, size, buffer_set, temp_info, heatdis->alpha, errorBound, iter, false, true);
                compressBlockPlaneFromPrePred(x, size, buffer_set, cmpkit_set, next, iter, false);
            }else{
                buffer_set->set_next_decmp_buffer_top(buffer_set->nextPlane_data_pos);
                recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set, current, iter);
                heatdisProcessBlockPlanePrePred(x, size, buffer_set, temp_info, heatdis->alpha, errorBound, iter, false, false);
                compressBlockPlaneFromPrePred(x, size, buffer_set, cmpkit_set, next, iter, false);
            }
        }
    }
}

template <class T>
inline void heatdisUpdatePrePred(
    DSize_3d& size,
    HeatDis3D *heatdis,
    TempInfo3D& temp_info,
    ht3DBufferSet *buffer_set,
    SZpCmpBufferSet *cmpkit_set,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        if(verb){
            if(iter >= ht3d_plot_offset && iter % ht3d_plot_gap == 0){
                SZp_decompress_3dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
                std::string h_name = heatdis3d_data_dir + "/h.L3.pre." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, size.nbEle);
                cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
                printf("prepred iter %d: cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / cmpSize);
                fflush(stdout);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePrePred(size, heatdis, temp_info, buffer_set, cmpkit_set, errorBound, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZp_decompress_3dLorenzo(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.dim3, size.Bsize, errorBound);
        std::string h_name = heatdis3d_data_dir + "/h.L3.pre." + std::to_string(iter);
        writefile(h_name.c_str(), h, size.nbEle);
        cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
        printf("prepred exit cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / cmpSize);
    }
    printf("prepred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdateDOC(
    DSize_3d& size,
    size_t dim1_padded,
    size_t dim2_padded,
    size_t dim3_padded,
    HeatDis3D *heatdis,
    TempInfo3D& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    size_t nbEle_padded = dim1_padded * dim2_padded * dim3_padded;
    T * h = (T *)malloc(nbEle_padded * sizeof(T));
    T * h2 = (T *)malloc(nbEle_padded * sizeof(T));
    unsigned char * compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    heatdis->initData(h, h2, temp_info.T_init);
    SZp_compress_3dLorenzo(h, compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, cmpSize);
    T * tmp = nullptr;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        clock_gettime(CLOCK_REALTIME, &start);
        SZp_decompress_3dLorenzo(h, compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter >= ht3d_plot_offset && iter % ht3d_plot_gap == 0){
                std::string h_name = heatdis3d_data_dir + "/h.L3.doc." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, nbEle_padded);
                printf("doc iter %d: cr = %.2f\n", iter-1, 1.0 * nbEle_padded * sizeof(T) / cmpSize);
                fflush(stdout);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdis->reset_source(h);
        heatdis->iterate(h, h2, tmp);
        SZp_compress_3dLorenzo(h, compressed, dim1_padded, dim2_padded, dim3_padded, size.Bsize, errorBound, cmpSize);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        std::string h_name = heatdis3d_data_dir + "/h.L3.doc." + std::to_string(iter);
        writefile(h_name.c_str(), h, nbEle_padded);
        printf("doc exit cr = %.2f\n", 1.0 * nbEle_padded * sizeof(T) / cmpSize);
    }
    printf("doc elapsed_time = %.6f\n", elapsed_time);
    free(compressed);
    free(h);
    free(h2);
}

template <class T>
void SZp_heatdis_3dLorenzo(
    unsigned char *cmpDataBuffer,
    size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, int max_iter,
    float T_top, float T_bott,
    float T_wall, float T_init,
    float alpha, double errorBound,
    decmpState state, bool verb
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t buffer_dim1 = size.Bsize + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    size_t dim1_padded = size.dim1 + 2;
    size_t nbEle_padded = dim1_padded * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 4 * sizeof(int));
    int * Buffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * 2 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * signPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));

    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        cmpData[i] = (unsigned char *)malloc(nbEle_padded * sizeof(T));
        offsets[i] = (int *)malloc(size.block_dim1 * sizeof(int));
    }
    memcpy(cmpData[0], cmpDataBuffer, nbEle_padded * sizeof(T));

    size_t prefix_length = 0;
    int block_index = 0;

    for(size_t x=0; x<size.block_dim1; x++){
        offsets[0][x] = prefix_length;
        offsets[1][x] = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim2; z++){
                int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
                int size_y = ((y+1) * size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y * size.Bsize;
                int size_z = ((z+1) * size.Bsize < size.dim2) ? size.Bsize : size.dim2 - z * size.Bsize;
                int block_size = size_x * size_y * size_z;
                int cmp_block_sign_length = (block_size + 7) / 8;
                int fixed_rate = (int)cmpDataBuffer[block_index];
                size_t Bytes = compute_encoding_byteLength(block_size, fixed_rate);
                if(fixed_rate)
                    prefix_length += (cmp_block_sign_length + Bytes);
                block_index++;
            }
        }
    }

    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, offsets, absPredError, signPredError, signFlag);
    ht3DBufferSet * buffer_set = new ht3DBufferSet(dim1, dim2, dim3, Buffer_3d, Buffer_2d);
    HeatDis3D * heatdis = new HeatDis3D(T_top, T_bott, T_wall, alpha, dim1, dim2, dim3);
    TempInfo3D temp_info(T_top, T_bott, T_wall, T_init, errorBound);

    switch(state){
        case decmpState::full:{
            heatdisUpdateDOC<T>(size, dim1_padded, buffer_dim2, buffer_dim3, heatdis, temp_info, errorBound, max_iter, verb);
            break;
        }
        case decmpState::prePred:{
            heatdisUpdatePrePred<T>(size, heatdis, temp_info, buffer_set, cmpkit_set, errorBound, max_iter, verb);
            break;
        }
        case decmpState::postPred:{
            break;
        }
    }

    delete(cmpkit_set);
    delete(buffer_set);
    delete(heatdis);
    free(Buffer_3d);
    free(Buffer_2d);
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
void SZp_heatdis_3dLorenzo(
    unsigned char *cmpDataBuffer, ht3DSettings& s, decmpState state, bool verb
){
    SZp_heatdis_3dLorenzo<T>(cmpDataBuffer, s.dim1, s.dim2, s.dim3, s.B, s.steps, s.T_top, s.T_bott, s.T_wall, s.T_init, s.alpha, s.eb, state, verb);
}

#endif