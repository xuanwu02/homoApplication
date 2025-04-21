#ifndef _SZ_APP_UTILS_HPP
#define _SZ_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include "SZ_def.hpp"
#include "utils.hpp"

struct CmpBufferSet
{
    unsigned char * compressed;
    unsigned int * absPredError;
    unsigned char * signFlag;
    int * mean_quant_inds;
    CmpBufferSet(
    unsigned char *cmpData_,
    unsigned int *absPredError_,
    unsigned char *signFlag_,
    int * mean_)
        : compressed(cmpData_),
          absPredError(absPredError_),
          signFlag(signFlag_),
          mean_quant_inds(mean_)
    {}
};

struct AppBufferSet2D_1d
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_size;
    size_t offset_0;
    int * buffer_2d;
    int * prevBlockSlice;
    int * currBlockSlice;
    int * nextBlockSlice;
    int * currSlice_data_pos;
    int * prevSlice_data_pos;
    int * nextSlice_data_pos;
    AppBufferSet2D_1d(
    size_t dim1, size_t dim2, int *buffer_2d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_2d(buffer_2d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2;
        offset_0 = buffer_dim2;
        prevBlockSlice = buffer_2d;
        currBlockSlice = buffer_2d + buffer_size;
        nextBlockSlice = buffer_2d + 2 * buffer_size;
    }
    void reset(){
        currSlice_data_pos = currBlockSlice + offset_0;
        prevSlice_data_pos = prevBlockSlice + offset_0;
        nextSlice_data_pos = nextBlockSlice + offset_0;
    }
    inline void setGhostEle(DSize2D_1d& size, bool isTopSlice, bool isBottomSlice){
        if(!isTopSlice){
            int * prevBlockSliceBottom_pos = prevSlice_data_pos + (size.Bwidth - 1) * offset_0;
            memcpy(currSlice_data_pos - offset_0, prevBlockSliceBottom_pos, offset_0 * sizeof(int));
        }
        if(!isBottomSlice){
            int * nextBlockSliceTop_pos = nextSlice_data_pos;
            memcpy(currSlice_data_pos + size.Bwidth * offset_0, nextBlockSliceTop_pos, offset_0 * sizeof(int));
        }
    }
};

struct AppBufferSet3D_1d
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t offset_0;
    size_t offset_1;
    int * buffer_3d;
    int * prevBlockSlice;
    int * currBlockSlice;
    int * nextBlockSlice;
    int * currSlice_data_pos;
    int * prevSlice_data_pos;
    int * nextSlice_data_pos;
    AppBufferSet3D_1d(
    size_t dim1, size_t dim2, size_t dim3, int *buffer_3d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        offset_0 = buffer_dim2 * buffer_dim3;
        offset_1 = buffer_dim3;
        prevBlockSlice = buffer_3d;
        currBlockSlice = buffer_3d + buffer_size;
        nextBlockSlice = buffer_3d + 2 * buffer_size;
    }
    void reset(){
        currSlice_data_pos = currBlockSlice + offset_0;
        prevSlice_data_pos = prevBlockSlice + offset_0;
        nextSlice_data_pos = nextBlockSlice + offset_0;
    }
    inline void setGhostEle(DSize3D_1d& size, bool isTopSlice, bool isBottomSlice){
        if(!isTopSlice){
            int * prevBlockSliceBottom_pos = prevSlice_data_pos + (size.Bwidth - 1) * offset_0;
            memcpy(currSlice_data_pos-offset_0, prevBlockSliceBottom_pos, offset_0*sizeof(int));
        }
        if(!isBottomSlice){
            int * nextBlockSliceTop_pos = nextSlice_data_pos;
            memcpy(currSlice_data_pos+size.Bwidth*offset_0, nextBlockSliceTop_pos, offset_0*sizeof(int));
        }
    }
};


/**
 * SZp stuff
*/

struct AppBufferSet_2d
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_size;
    size_t offset_0;
    int * buffer_2d;
    int * prevBlockSlice;
    int * currBlockSlice;
    int * nextBlockSlice;
    int * dy_buffer;
    int * decmp_buffer;
    int * currSlice_data_pos;
    int * prevSlice_data_pos;
    int * nextSlice_data_pos;
    AppBufferSet_2d(
    size_t dim1, size_t dim2, int *buffer_2d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_2d(buffer_2d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2;
        offset_0 = buffer_dim2;
        prevBlockSlice = buffer_2d;
        currBlockSlice = buffer_2d + buffer_size;
        nextBlockSlice = buffer_2d + 2 * buffer_size;
        dy_buffer = buffer_2d + 3 * buffer_size;
        decmp_buffer = buffer_2d + 3 * buffer_size + offset_0;
    }
    void reset(){
        currSlice_data_pos = currBlockSlice + offset_0 + 1;
        prevSlice_data_pos = prevBlockSlice + offset_0 + 1;
        nextSlice_data_pos = nextBlockSlice + offset_0 + 1;
    }
    inline void setGhostEle(DSize_2d& size, bool isTopSlice, bool isBottomSlice){
        if(!isTopSlice){
            int * prevBlockSliceBottom_pos = isTopSlice ? nullptr : prevSlice_data_pos + (size.Bsize - 1) * offset_0 - 1;
            memcpy(currSlice_data_pos-offset_0-1, prevBlockSliceBottom_pos, offset_0*sizeof(int));
        }
        if(!isBottomSlice){
            int * nextBlockSliceTop_pos = isBottomSlice ? nullptr : nextSlice_data_pos - 1;
            memcpy(currSlice_data_pos+size.Bsize*offset_0-1, nextBlockSliceTop_pos, offset_0*sizeof(int));
        }
    }
};

struct AppBufferSet_3d
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t offset_0;
    size_t offset_1;
    int * buffer_3d;
    int * prevBlockSlice;
    int * currBlockSlice;
    int * nextBlockSlice;
    int * decmp_buffer;
    int * currSlice_data_pos;
    int * prevSlice_data_pos;
    int * nextSlice_data_pos;
    AppBufferSet_3d(
    size_t dim1, size_t dim2, size_t dim3,
    int *buffer_3d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        offset_0 = buffer_dim2 * buffer_dim3;
        offset_1 = buffer_dim3;
        prevBlockSlice = buffer_3d;
        currBlockSlice = buffer_3d + buffer_size;
        nextBlockSlice = buffer_3d + 2 * buffer_size;
        decmp_buffer = buffer_3d + 3 * buffer_size;
    }
    void reset(){
        currSlice_data_pos = currBlockSlice + offset_0 + offset_1 + 1;
        prevSlice_data_pos = prevBlockSlice + offset_0 + offset_1 + 1;
        nextSlice_data_pos = nextBlockSlice + offset_0 + offset_1 + 1;
        memset(buffer_3d, 0, 4 * buffer_size * sizeof(int));
    }
    inline void setGhostEle(DSize_3d& size, bool isTopSlice, bool isBottomSlice){
        if(!isTopSlice){
            int * prevBlockSliceBottom_pos = prevSlice_data_pos + (size.Bsize - 1) * offset_0 - offset_1 - 1;
            memcpy(currSlice_data_pos-offset_0-offset_1-1, prevBlockSliceBottom_pos, offset_0*sizeof(int));
        }
        if(!isBottomSlice){
            int * nextBlockSliceTop_pos = nextSlice_data_pos - offset_1 - 1;
            memcpy(currSlice_data_pos+size.Bsize*offset_0-offset_1-1, nextBlockSliceTop_pos, offset_0*sizeof(int));
        }
    }
};

inline int* allocateAndZero1D(size_t n) {
    int* ptr = new int[n]();  
    return ptr;
}

inline int** allocateAndZero2D(size_t rows, size_t cols) {
    int** arr = new int*[rows];
    for(size_t r=0; r<rows; r++) {
        arr[r] = new int[cols]();
    }
    return arr;
}

struct derivIntBuffer_3d
{
    int * dx_buffer;
    int ** dy_buffer;
    int ** dz_buffer;
    derivIntBuffer_3d(int *px, int **py, int **pz){
        dx_buffer = px;
        dy_buffer = py;
        dz_buffer = pz;
    }
};

template <class T>
inline int predict_lorenzo_2d(
    const T *data_pos, int *buffer_pos,
    size_t offset_0, double inver_eb
){
    int curr_quant = SZ_quantize(data_pos[0], inver_eb);
    buffer_pos[0] = curr_quant;
    return curr_quant - buffer_pos[-1] - buffer_pos[-offset_0] + buffer_pos[-offset_0-1];
}

inline void recover_lorenzo_2d(
    int *buffer_pos, size_t offset_0
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-offset_0] - buffer_pos[-offset_0-1]);
}

template <class T>
inline int predict_lorenzo_3d(
    const T *data_pos, int *buffer_pos, double inver_eb,
    size_t offset_0, size_t offset_1
){
    int curr_quant = SZ_quantize(data_pos[0], inver_eb);
    buffer_pos[0] = curr_quant;
    int pred = buffer_pos[-1] + buffer_pos[-offset_1] + buffer_pos[-offset_0] 
            - buffer_pos[-offset_1 - 1] - buffer_pos[-offset_0 - 1] 
            - buffer_pos[-offset_0 - offset_1] + buffer_pos[-offset_0 - offset_1 - 1];
    return curr_quant - pred;
}

inline void recover_lorenzo_3d(
    int *buffer_pos, size_t offset_0, size_t offset_1
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-offset_1] + buffer_pos[-offset_0] 
                    - buffer_pos[-offset_1 - 1] - buffer_pos[-offset_0 - 1] 
                    - buffer_pos[-offset_0 - offset_1] + buffer_pos[-offset_0 - offset_1 - 1];
}

template <class T>
inline void deriv_lorenzo_2d(
    const int *pred_level_0_pos, const int *pred_level_1_pos,
    int *res_int_buffer_pos, T *res_pos, size_t res_offset_1, double errorBound
){
    int res_integer = pred_level_0_pos[0] + pred_level_1_pos[0]
                    + res_int_buffer_pos[-1] + res_int_buffer_pos[-res_offset_1] - res_int_buffer_pos[-res_offset_1-1];
    res_int_buffer_pos[0] = res_integer;
    res_pos[0] = res_integer * errorBound;
}


/**
 * SZx stuff
*/

template <class T>
inline void dxdy_compute_block_mean_difference(
    int x, int block_dim2, double errorBound, int *block_mean_buffer,
    T *dx_top_blockmean_diff, T *dx_bott_blockmean_diff, T *dy_blockmean_diff
){
    int * currRow_mean_quant = block_mean_buffer + x * block_dim2;
    int * prevRow_mean_quant = currRow_mean_quant - block_dim2;
    int * nextRow_mean_quant = currRow_mean_quant + block_dim2;
    dy_blockmean_diff[0] = 0;
    dy_blockmean_diff[block_dim2+2] = 0;
    for(int j=0; j<block_dim2; j++){
        dy_blockmean_diff[j+1] = (currRow_mean_quant[j+1] - currRow_mean_quant[j]) * errorBound;
        dx_top_blockmean_diff[j] = (currRow_mean_quant[j] - prevRow_mean_quant[j]) * errorBound;
        dx_bott_blockmean_diff[j] = (nextRow_mean_quant[j] - currRow_mean_quant[j]) * errorBound;
    }
}

template <class T>
inline void laplacian_compute_block_mean_difference(
    int x, int block_dim2, double errorBound, int *block_mean_buffer,
    T *dx_top_blockmean_diff, T *dx_bott_blockmean_diff, T *dy_blockmean_diff
){
    int * currRow_mean_quant = block_mean_buffer + x * block_dim2;
    int * prevRow_mean_quant = currRow_mean_quant - block_dim2;
    int * nextRow_mean_quant = currRow_mean_quant + block_dim2;
    dy_blockmean_diff[0] = 0;
    dy_blockmean_diff[block_dim2+2] = 0;
    for(int j=0; j<block_dim2; j++){
        dy_blockmean_diff[j+1] = (currRow_mean_quant[j+1] - currRow_mean_quant[j]) * errorBound * 2;
        dx_top_blockmean_diff[j] = (currRow_mean_quant[j] - prevRow_mean_quant[j]) * errorBound * 2;
        dx_bott_blockmean_diff[j] = (nextRow_mean_quant[j] - currRow_mean_quant[j]) * errorBound * 2;
    }
}

// 1d compress
template <class T>
inline int compute_block_mean_quant(
    int block_size, const T *data_pos, int *block_buffer, double inver_eb
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * curr_data_pos = data_pos;
    for(int i=0; i<block_size; i++){
        int curr_quant = SZ_quantize(*curr_data_pos++, inver_eb);
        *block_buffer_pos++ = curr_quant;
        sum += curr_quant;
    }
    int mean_quant = sum / block_size;
    mean_quant -= (mean_quant < 0) ? 1 : 0;
    return mean_quant;
}

// 2d compress
template <class T>
inline int compute_block_mean_quant(
    int size_x, int size_y, size_t dim0_offset,
    const T *data_pos, int *block_buffer, double inver_eb
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * curr_data_pos = data_pos;
    for(int i=0; i<size_x; i++){
        for(int j=0; j<size_y; j++){
            int curr_quant = SZ_quantize(*curr_data_pos++, inver_eb);
            *block_buffer_pos++ = curr_quant;
            sum += curr_quant;
        }
        curr_data_pos += dim0_offset - size_y;
    }
    int mean_quant = sum / (size_x * size_y);
    mean_quant -= (mean_quant < 0) ? 1 : 0;
    return mean_quant;
}

// 3d compress
template <class T>
inline int compute_block_mean_quant(
    int size_x, int size_y, int size_z, size_t dim0_offset,
    size_t dim1_offset, const T *data_pos, int *block_buffer, double inver_eb
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * x_data_pos = data_pos;
    for(int i=0; i<size_x; i++){
        const T * y_data_pos = x_data_pos;
        for(int j=0; j<size_y; j++){
            const T * z_data_pos = y_data_pos;
            for(int k=0; k<size_z; k++){
                int curr_quant = SZ_quantize(*z_data_pos++, inver_eb);
                *block_buffer_pos++ = curr_quant;
                sum += curr_quant;
            }
            y_data_pos += dim1_offset;
        }
        x_data_pos += dim0_offset;
    }
    int mean_quant = sum / (size_x * size_y * size_z);
    mean_quant -= (mean_quant < 0) ? 1 : 0;
    return mean_quant;
}

// decompress
inline void extract_block_mean(
    unsigned char *cmpData_pos, int *blocks_mean_quant, size_t num_blocks
){
    unsigned char * qmean_pos = cmpData_pos;
    for(size_t k=0; k<num_blocks; k++){
        // memcpy(&blocks_mean_quant[k], qmean_pos, sizeof(int));
        memcpy(blocks_mean_quant+k, qmean_pos, sizeof(int));
        qmean_pos += 4;
    }
}

template <class T>
inline T compute_integer_mean_2d(
    DSize_2d& size, const unsigned char *cmpData_pos, int *blocks_mean_quant
){
    const unsigned char * qmean_pos = cmpData_pos;
    int block_ind = 0;
    T sum = 0;
    int mean;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            memcpy(&mean, qmean_pos, sizeof(int));
            blocks_mean_quant[block_ind++] = mean;
            qmean_pos += 4;
            sum += mean * block_size;
        }
    }
    T int_mean = sum / (T)size.nbEle;
    return int_mean;
}

template <class T>
inline T compute_integer_mean_3d(
    DSize_3d& size, const unsigned char *cmpData_pos, int *blocks_mean_quant
){
    const unsigned char * qmean_pos = cmpData_pos;
    int block_ind = 0;
    T sum = 0;
    int mean;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                memcpy(&mean, qmean_pos, sizeof(int));
                blocks_mean_quant[block_ind++] = mean;
                qmean_pos += 4;
                sum += mean * block_size;
            }
        }
    }
    T int_mean = sum / (T)size.nbEle;
    return int_mean;
}

#endif