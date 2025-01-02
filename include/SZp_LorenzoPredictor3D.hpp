#ifndef _SZP_HEATDIS_3DLORENZO_HPP
#define _SZP_HEATDIS_3DLORENZO_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "typemanager.hpp"
#include "SZp_app_utils.hpp"

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
                            quant_diff = predict_lorenzo_3d(curr_data_pos, curr_buffer_pos, errorBound, buffer_dim0_offset, buffer_dim1_offset);
                            curr_data_pos++;
                            curr_buffer_pos++;
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
    bool flag = true;
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
                            recover_lorenzo_3d(quant_sum, curr_buffer_pos, buffer_dim0_offset, buffer_dim1_offset);
                            curr_buffer_pos++;
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

struct timespec start, end;
double postPred_decmp_time = 0;
double postPred_op_time = 0;
double postPred_cmp_time = 0;
double prePred_decmp_time = 0;
double prePred_op_time = 0;
double prePred_cmp_time = 0;

inline void recoverBlockPlane2PostPred(
    size_t x, DSize_3d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d *buffer_set, SZpCmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start);
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
clock_gettime(CLOCK_REALTIME, &end);
postPred_decmp_time += (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
}

template <class T>
inline void dxdydzProcessBlockPlanePostPred(
    size_t x, DSize_3d size, derivIntBuffer_3d *deriv_buffer,
    SZpAppBufferSet_3d *buffer_set, double errorBound,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start);
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
clock_gettime(CLOCK_REALTIME, &end);
postPred_op_time += (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
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
clock_gettime(CLOCK_REALTIME, &start);
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
clock_gettime(CLOCK_REALTIME, &end);
prePred_decmp_time += (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred(
    size_t x, DSize_3d& size, SZpAppBufferSet_3d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start);
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
clock_gettime(CLOCK_REALTIME, &end);
prePred_op_time += (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
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
    size_t buffer_dim1 = size.Bsize + 1;
    size_t buffer_dim2 = size.dim2 + 1;
    size_t buffer_dim3 = size.dim3 + 1;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * Buffer_2d = (int *)malloc(buffer_dim2 * buffer_dim3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_3d * buffer_set = new SZpAppBufferSet_3d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, Buffer_2d, appType::CENTRALDIFF);
    SZpCmpBufferSet * cmpkit_set = new SZpCmpBufferSet(cmpData, signPredError, signFlag);
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    T * dz_pos = dz_result;
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
    }
    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_3d);
    free(Buffer_2d);
    free(signPredError);
    free(signFlag);
}

#endif