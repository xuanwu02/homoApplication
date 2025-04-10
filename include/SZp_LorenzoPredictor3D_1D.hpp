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
#include "settings.hpp"

struct timespec start2, end2;
double rec_time = 0;
double op_time = 0;

template <class T>
void SZp_compress3D_1dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    double inver_eb = 0.5 / errorBound;
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim3+1)*sizeof(int));
    // memset(quant_buffer, 0, (size.dim3+1)*sizeof(int));
    quant_buffer[0] = 0;
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            const T * z_data_pos = y_data_pos;
            int * buffer_start_pos = quant_buffer + 1;
            for(size_t z=0; z<size.block_dim3; z++){
                int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                unsigned int * abs_err_pos = absPredError;
                unsigned char * sign_pos = signFlag;
                int max_err = 0;
                int * block_buffer_pos = buffer_start_pos;
                const T * curr_data_pos = z_data_pos;
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
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.dim1_offset;
        }
        x_data_pos += size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(quant_buffer);
    free(absPredError);
    free(signFlag);
}

template <class T>
void SZp_decompress3D_1dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength,
    double errorBound
){
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim3+1)*sizeof(int));
    // memset(quant_buffer, 0, (size.dim3+1)*sizeof(int));
    quant_buffer[0] = 0;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * encode_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            T * z_data_pos = y_data_pos;
            int * buffer_start_pos = quant_buffer + 1;
            for(size_t z=0; z<size.block_dim3; z++){
                int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int fixed_rate = (int)cmpData[block_ind++];
                int * block_buffer_pos = buffer_start_pos;
                T * curr_data_pos = z_data_pos;
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
                z_data_pos += size.Bsize;
            }
            y_data_pos += size.dim1_offset;
        }
        x_data_pos += size.dim0_offset;
    }
}

double SZp_mean3D_1dLorenzo_recover2PostPred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            size_t offset = 0;
            for(size_t z=0; z<size.block_dim3; z++){
                int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
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
                    quant_sum += (size.dim3 - (offset + j)) * signPredError[j];
                }
                offset += block_size;
            }
        }
    }
    free(signPredError);
    free(signFlag);
    double mean = 2 * errorBound * (double)quant_sum / size.nbEle;
    return mean;
}

template <class T>
double SZp_mean3D_1dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress3D_1dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    return mean;
}

template <class T>
double SZp_mean3D_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound, decmpState state
){
    double mean = -9999;
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::postPred:{
            mean = SZp_mean3D_1dLorenzo_recover2PostPred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::prePred:{
            // mean = SZp_mean3D_1dLorenzo_recover2PrePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
            break;
        }
        case decmpState::full:{
            mean = SZp_mean3D_1dLorenzo_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return mean;
}

double SZp_variance3D_1dLorenzo_recover2PrePred(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound
){
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    int * quant_buffer = (int *)malloc((size.dim3+1)*sizeof(int));
    quant_buffer[0] = 0;
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    uint64_t squared_quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        for(size_t y=0; y<size.block_dim2; y++){
            int * data_pos = quant_buffer + 1;
            for(size_t z=0; z<size.block_dim3; z++){
                uint64_t block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
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
    }
    free(quant_buffer);
    free(signPredError);
    free(signFlag);
    double var = (2 * errorBound) * (2 * errorBound)* ((double)squared_quant_sum - (double)quant_sum * quant_sum / size.nbEle) / (size.nbEle - 1);
    return var;
}

template <class T>
double SZp_variance3D_1dLorenzo_decOp(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    T *decData, int blockSideLength, double errorBound
){
    size_t nbEle = dim1 * dim2 * dim3;
    SZp_decompress3D_1dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
    double mean = 0;
    for(size_t i=0; i<nbEle; i++) mean += decData[i];
    mean /= nbEle;
    double var = 0;
    for(size_t i=0; i<nbEle; i++) var += (decData[i] - mean) * (decData[i] - mean);
    var /= (nbEle - 1);
    return var;
}

template <class T>
double SZp_variance3D_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3, T *decData,
    int blockSideLength, double errorBound, decmpState state
){
    double var = -9999;

    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_REALTIME, &start);
    switch(state){
        case decmpState::full:{
            var = SZp_variance3D_1dLorenzo_decOp(cmpData, dim1, dim2, dim3, decData, blockSideLength, errorBound);            
            break;
        }
        case decmpState::prePred:{
            var = SZp_variance3D_1dLorenzo_recover2PrePred(cmpData, dim1, dim2, dim3, blockSideLength, errorBound);            
            break;
        }
        case decmpState::postPred:{
            // var = SZp_variance_2dLorenzo_postPredMean(cmpData, dim1, dim2, blockSideLength, errorBound);            
            break;
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed_time = get_elapsed_time(start, end);
    printf("elapsed_time = %.6f\n", elapsed_time);

    return var;
}

inline void recoverBlockPlane2PostPred(
    size_t x, DSize_3d1d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d1d *buffer_set, SZpCmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.Bwidth * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int * block_buffer_pos = buffer_start_pos;
                int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
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
            buffer_start_pos += buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
        buffer_start_pos += buffer_set->buffer_dim0_offset - size.dim2 * buffer_set->buffer_dim1_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

// inline void recoverBlockPlane2PrePred2(
//     size_t x, DSize_3d1d& size, unsigned char *& encode_pos, int *buffer_data_pos,
//     SZpAppBufferSet_3d1d *buffer_set, SZpCmpBufferSet *cmpkit_set
// ){
//     int block_ind = x * size.Bwidth * size.block_dim2 * size.block_dim3;
//     int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
//     int * buffer_plane_pos = buffer_data_pos;
//     for(int i=0; i<size_x; i++){
//         int * buffer_row_pos = buffer_plane_pos;
//         for(size_t y=0; y<size.block_dim2; y++){
//             int * block_buffer_pos = buffer_row_pos;
//             for(size_t z=0; z<size.block_dim3; z++){
//                 int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
//                 int * curr_buffer_pos = block_buffer_pos;
//                 int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
//                 if(fixed_rate){
//                     size_t cmp_block_sign_length = (block_size + 7) / 8;
//                     convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
//                     encode_pos += cmp_block_sign_length;
//                     unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
//                     encode_pos += savedbitsbytelength;
//                     convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
//                 }else{
//                     for(int j=0; j<block_size; j++){
//                         cmpkit_set->signPredError[j] = 0;
//                     }
//                 }
//                 for(int j=0; j<block_size; j++){
//                     curr_buffer_pos[0] = cmpkit_set->signPredError[j];
//                     recover_lorenzo_1d(curr_buffer_pos++);
//                 }
//                 block_buffer_pos += size.Bsize;
//             }
//             buffer_row_pos += buffer_set->buffer_dim1_offset;
//         }
//         buffer_plane_pos += buffer_set->buffer_dim0_offset;
//     }
// }

inline void recoverBlockPlane2PrePred(
    size_t x, DSize_3d1d& size, unsigned char *& encode_pos, int *buffer_data_pos,
    SZpAppBufferSet_3d1d *buffer_set, SZpCmpBufferSet *cmpkit_set
){
clock_gettime(CLOCK_REALTIME, &start2);
    int block_ind = x * size.Bwidth * size.block_dim2 * size.block_dim3;
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            for(size_t z=0; z<size.block_dim3; z++){
                int block_size = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int * block_buffer_pos = buffer_start_pos;
                int fixed_rate = (int)cmpkit_set->compressed[block_ind++];
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
            buffer_start_pos += buffer_set->buffer_dim1_offset - size.Bsize * size.block_dim3;
        }
        buffer_start_pos += buffer_set->buffer_dim0_offset - size.dim2 * buffer_set->buffer_dim1_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
rec_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlockPlanePrePred(
    size_t x, DSize_3d1d& size, SZpAppBufferSet_3d1d *buffer_set,
    T *dx_start_pos, T *dy_start_pos, T *dz_start_pos,
    double errorBound, bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
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
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_dx_pos[k] = (next_plane[buffer_index_2d] - prev_plane[buffer_index_2d]) * errorBound;
                y_dy_pos[k] = (next_row[k] - prev_row[k]) * errorBound;
                y_dz_pos[k] = (curr_row[k + 1] - curr_row[k - 1]) * errorBound;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void dxdydzProcessBlocksPrePred(
    DSize_3d1d &size,
    size_t numBlockPlane,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_3d1d *buffer_set,
    unsigned char *&encode_pos,
    T *dx_pos, T *dy_pos, T *dz_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bwidth * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockPlane = nullptr;
    for(size_t x=0; x<numBlockPlane; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set);
            recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
            dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == numBlockPlane - 1){
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, true);
            }else{
                recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
                dxdydzProcessBlockPlanePrePred(x, size, buffer_set, dx_pos+offset, dy_pos+offset, dz_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_dxdydz_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound, T *dx_result,
    T *dy_result, T *dz_result, decmpState state
){
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    size_t numblockPlane = (size.dim1 - 1) / size.Bwidth + 1;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_3d1d * buffer_set = new SZpAppBufferSet_3d1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
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
            break;
        }
        case decmpState::prePred:{
            dxdydzProcessBlocksPrePred(size, numblockPlane, cmpkit_set, buffer_set, encode_pos, dx_pos, dy_pos, dz_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress3D_1dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(decData);
}

template <class T>
inline void laplacianProcessBlockPlanePrePred(
    size_t x, DSize_3d1d& size, SZpAppBufferSet_3d1d *buffer_set,
    T *result_start_pos, double errorBound,
    bool isTopPlane, bool isBottomPlane
){
clock_gettime(CLOCK_REALTIME, &start2);
    int size_x = ((x+1)*size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    const int * prevBlockPlaneBottom_pos = buffer_set->prevPlane_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - buffer_set->buffer_dim1_offset - 1;
    const int * nextBlockPlaneTop_pos = buffer_set->nextPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    if(!isTopPlane) memcpy(buffer_set->currPlane_data_pos-buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, prevBlockPlaneBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomPlane) memcpy(buffer_set->currPlane_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-buffer_set->buffer_dim1_offset-1, nextBlockPlaneTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    const int * curr_plane = buffer_set->currPlane_data_pos - buffer_set->buffer_dim1_offset - 1;
    for(int i=0; i<size_x; i++){
        T * laplacian_pos = result_start_pos + i * size.dim0_offset;
        const int * prev_plane = curr_plane - buffer_set->buffer_dim0_offset;
        const int * next_plane = curr_plane + buffer_set->buffer_dim0_offset;
        const int * curr_row = curr_plane + buffer_set->buffer_dim1_offset + 1;
        for(size_t j=0; j<size.dim2; j++){
            T * y_laplacian_pos = laplacian_pos + j * size.dim1_offset;
            const int * prev_row = curr_row - buffer_set->buffer_dim1_offset;
            const int * next_row = curr_row + buffer_set->buffer_dim1_offset;
            for(size_t k=0; k<size.dim3; k++){
                size_t index_1d = k;
                size_t buffer_index_2d = (j + 1) * buffer_set->buffer_dim1_offset + k + 1;
                y_laplacian_pos[index_1d] = (curr_row[k-1] + curr_row[k+1] +
                                             prev_row[k] + next_row[k] +
                                             prev_plane[buffer_index_2d] + next_plane[buffer_index_2d] -
                                             6 * curr_row[k]) * errorBound * 2;
            }
            curr_row += buffer_set->buffer_dim1_offset;
        }
        curr_plane += buffer_set->buffer_dim0_offset;
    }
clock_gettime(CLOCK_REALTIME, &end2);
op_time += get_elapsed_time(start2, end2);
}

template <class T>
inline void laplacianProcessBlocksPrePred(
    DSize_3d1d &size,
    size_t numBlockPlane,
    SZpCmpBufferSet *cmpkit_set, 
    SZpAppBufferSet_3d1d *buffer_set,
    unsigned char *&encode_pos,
    T *result_pos,
    double errorBound
){
    size_t BlockPlaneSize = size.Bwidth * size.dim2 * size.dim3;
    buffer_set->reset();
    int * tempBlockPlane = nullptr;
    for(size_t x=0; x<numBlockPlane; x++){
        size_t offset = x * BlockPlaneSize;
        if(x == 0){
            recoverBlockPlane2PrePred(x, size, encode_pos, buffer_set->currPlane_data_pos, buffer_set, cmpkit_set);
            recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
            laplacianProcessBlockPlanePrePred(x, size, buffer_set, result_pos+offset, errorBound, true, false);
        }else{
            rotate_buffer(buffer_set->currPlane_data_pos, buffer_set->prevPlane_data_pos, buffer_set->nextPlane_data_pos, tempBlockPlane);
            if(x == numBlockPlane - 1){
                laplacianProcessBlockPlanePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, true);
            }else{
                recoverBlockPlane2PrePred(x+1, size, encode_pos, buffer_set->nextPlane_data_pos, buffer_set, cmpkit_set);
                laplacianProcessBlockPlanePrePred(x, size, buffer_set, result_pos+offset, errorBound, false, false);
            }
        }
    }
    printf("recover_time = %.6f\n", rec_time);
    printf("process_time = %.6f\n", op_time);
}

template <class T>
void SZp_laplacian3D_1dLorenzo(
    unsigned char *cmpData, size_t dim1, size_t dim2, size_t dim3,
    int blockSideLength, double errorBound,
    T *laplacian_result, decmpState state
){
    DSize_3d1d size(dim1, dim2, dim3, blockSideLength);
    size_t numblockPlane = (size.dim1 - 1) / size.Bwidth + 1;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_dim3 = size.dim3 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
    int * Buffer_3d = (int *)malloc(buffer_size * 3 * sizeof(int));
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    T * decData = (T *)malloc(size.nbEle * sizeof(T));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    SZpAppBufferSet_3d1d * buffer_set = new SZpAppBufferSet_3d1d(buffer_dim1, buffer_dim2, buffer_dim3, Buffer_3d, appType::CENTRALDIFF);
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
            laplacianProcessBlocksPrePred(size, numblockPlane, cmpkit_set, buffer_set, encode_pos, laplacian_pos, errorBound);
            break;
        }
        case decmpState::full:{
            clock_gettime(CLOCK_REALTIME, &start2);
            SZp_decompress3D_1dLorenzo(decData, cmpData, dim1, dim2, dim3, blockSideLength, errorBound);
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
    free(signPredError);
    free(signFlag);
    free(decData);
}

#endif