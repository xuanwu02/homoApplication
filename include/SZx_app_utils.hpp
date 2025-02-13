#ifndef _SZX_APP_UTILS_HPP
#define _SZX_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <string.h>
#include <algorithm>
#include "SZ_def.hpp"

struct TempInfo2D
{
    float src_temp, wall_temp, init_temp, ratio;
    const int q_S;
    const int q_W;
    TempInfo2D(
    float src, float wall, float init, float ratio, double eb)
        : src_temp(src),
          wall_temp(wall),
          init_temp(init),
          ratio(ratio), 
          q_S(SZ_quantize(src, eb)),
          q_W(SZ_quantize(wall, eb))
    {}
    void prepare_src_row(size_t dim2, int *quant_buffer){
        size_t c1 =  dim2 * (1.0 - ratio) * 0.5 + 1;
        size_t c2 =  dim2 * (1.0 + ratio) * 0.5 - 1;
        size_t j;
        for(j=0; j<dim2; j++){
            quant_buffer[j] = q_S;
            if(j < c1 || j > c2) quant_buffer[j] = q_W;
        }
    }
};

struct SZxCmpBufferSet
{
    unsigned char ** cmpData;
    int ** offsets;
    unsigned char * compressed;
    int * mean_quant_inds;
    unsigned int * absPredError;
    int * signPredError;
    unsigned char * signFlag;
    size_t cmpSize;
    size_t prefix_length;
    SZxCmpBufferSet(
    unsigned char *cmpData_, int *mean_,
    int *signPredError_, unsigned char *signFlag_)
        : compressed(cmpData_),
          mean_quant_inds(mean_),
          signPredError(signPredError_),
          signFlag(signFlag_)
    {}
    SZxCmpBufferSet(
    unsigned char **cmpData_, int **offsets_, int *mean_,
    unsigned int *absPredError_, int *signPredError_, unsigned char *signFlag_)
        : cmpData(cmpData_),
          offsets(offsets_),
          mean_quant_inds(mean_),
          absPredError(absPredError_),
          signPredError(signPredError_),
          signFlag(signFlag_)
    {}
    void reset(){
        cmpSize = 0;
        prefix_length = 0;
    }
};

struct SZxAppBufferSet_2d
{
    appType type;
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    int * buffer_2d;
    int * prevBlockRow;
    int * currBlockRow;
    int * nextBlockRow;
    int * updateBlockRow;
    int * updateRow_data_pos;
    int * currRow_data_pos;
    int * prevRow_data_pos;
    int * nextRow_data_pos;
    SZxAppBufferSet_2d(
    size_t dim1, size_t dim2, int *buffer_2d_, appType type_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_2d(buffer_2d_),
          type(type_)
    {
        buffer_size = buffer_dim1 * buffer_dim2;
        buffer_dim0_offset = buffer_dim2;
        prevBlockRow = buffer_2d;
        currBlockRow = buffer_2d + buffer_size;
        nextBlockRow = buffer_2d + 2 * buffer_size;
        switch(type){
            case appType::HEATDIS:{
                updateBlockRow = buffer_2d + 3 * buffer_size;
                break;
            }
            case appType::CENTRALDIFF:{
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void reset(){
        switch(type){
            case appType::HEATDIS:{
                memset(buffer_2d, 0, 3 * buffer_size * sizeof(int));
                updateRow_data_pos = updateBlockRow + buffer_dim0_offset + 1;
                currRow_data_pos = currBlockRow + buffer_dim0_offset + 1;
                prevRow_data_pos = prevBlockRow + buffer_dim0_offset + 1;
                nextRow_data_pos = nextBlockRow + buffer_dim0_offset + 1;
                break;
            }
            case appType::CENTRALDIFF:{
                currRow_data_pos = currBlockRow + buffer_dim0_offset + 1;
                prevRow_data_pos = prevBlockRow + buffer_dim0_offset + 1;
                nextRow_data_pos = nextBlockRow + buffer_dim0_offset + 1;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
};

struct SZxAppBufferSet_3d
{
    appType type;
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    size_t buffer_dim1_offset;
    int * buffer_3d;
    int * prevBlockPlane;
    int * currBlockPlane;
    int * nextBlockPlane;
    int * currPlane_data_pos;
    int * prevPlane_data_pos;
    int * nextPlane_data_pos;
    SZxAppBufferSet_3d(
    size_t dim1, size_t dim2, size_t dim3, int *buffer_3d_, appType type_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_),
          type(type_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        buffer_dim0_offset = buffer_dim2 * buffer_dim3;
        buffer_dim1_offset = buffer_dim3;
        prevBlockPlane = buffer_3d;
        currBlockPlane = buffer_3d + buffer_size;
        nextBlockPlane = buffer_3d + 2 * buffer_size;
    }
    void reset(){
        currPlane_data_pos = currBlockPlane;
        prevPlane_data_pos = prevBlockPlane;
        nextPlane_data_pos = nextBlockPlane;
    }
};

template <class T>
inline void compute_block_mean_difference(
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
inline int compute_block_mean_quant(
    int block_size, const T *data_pos, int *block_buffer, double errorBound
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * curr_data_pos = data_pos;
    for(int i=0; i<block_size; i++){
        int curr_quant = SZ_quantize(*curr_data_pos++, errorBound);
        *block_buffer_pos++ = curr_quant;
        sum += curr_quant;
    }
    int mean_quant = sum / block_size;
    return mean_quant;
}

template <class T>
inline int compute_block_mean_quant(
    int size_x, int size_y, size_t dim0_offset,
    const T *data_pos, int *block_buffer, double errorBound
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * curr_data_pos = data_pos;
    for(int i=0; i<size_x; i++){
        for(int j=0; j<size_y; j++){
            int curr_quant = SZ_quantize(*curr_data_pos++, errorBound);
            *block_buffer_pos++ = curr_quant;
            sum += curr_quant;
        }
        curr_data_pos += dim0_offset - size_y;
    }
    int mean_quant = sum / (size_x * size_y);
    return mean_quant;
}

template <class T>
inline int compute_block_mean_quant(
    int size_x, int size_y, int size_z, size_t dim0_offset,
    size_t dim1_offset, const T *data_pos, int *block_buffer, double errorBound
){
    int64_t sum = 0;
    int * block_buffer_pos = block_buffer;
    const T * x_data_pos = data_pos;
    for(int i=0; i<size_x; i++){
        const T * y_data_pos = x_data_pos;
        for(int j=0; j<size_y; j++){
            const T * z_data_pos = y_data_pos;
            for(int k=0; k<size_z; k++){
                int curr_quant = SZ_quantize(*z_data_pos++, errorBound);
                *block_buffer_pos++ = curr_quant;
                sum += curr_quant;
            }
            y_data_pos += dim1_offset;
        }
        x_data_pos += dim0_offset;
    }
    int mean_quant = sum / (size_x * size_y * size_z);
    return mean_quant;
}

inline double compute_mean_2d(
    DSize_2d& size, const unsigned char *cmpData_pos, double errorBound
){
    const unsigned char * qmean_pos = cmpData_pos;
    int64_t sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean = (0xff000000 & (*qmean_pos << 24)) |
                        (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                        (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                        (0x000000ff & *(qmean_pos+3));
            qmean_pos += 4;
            sum += mean * block_size;
        }
    }
    double mean = sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
inline T compute_integer_mean_2d(
    DSize_2d& size, const unsigned char *cmpData_pos, int *blocks_mean_quant
){
    const unsigned char * qmean_pos = cmpData_pos;
    int block_ind = 0;
    T sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x * size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int mean = (0xff000000 & (*qmean_pos << 24)) |
                        (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                        (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                        (0x000000ff & *(qmean_pos+3));
            blocks_mean_quant[block_ind++] = mean;
            qmean_pos += 4;
            sum += mean * block_size;
        }
    }
    T int_mean = sum / (T)size.nbEle;
    return int_mean;
}

inline double compute_mean_3d(
    DSize_3d& size, const unsigned char *cmpData_pos, double errorBound
){
    const unsigned char * qmean_pos = cmpData_pos;
    int64_t sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int mean = (0xff000000 & (*qmean_pos << 24)) |
                            (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                            (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                            (0x000000ff & *(qmean_pos+3));
                qmean_pos += 4;
                sum += mean * block_size;
            }
        }
    }
    double mean = sum * 2 * errorBound / size.nbEle;
    return mean;
}

template <class T>
inline T compute_integer_mean_3d(
    DSize_3d& size, const unsigned char *cmpData_pos, int *blocks_mean_quant
){
    const unsigned char * qmean_pos = cmpData_pos;
    int block_ind = 0;
    T sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1) * size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int mean = (0xff000000 & (*qmean_pos << 24)) |
                            (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                            (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                            (0x000000ff & *(qmean_pos+3));
                blocks_mean_quant[block_ind++] = mean;
                qmean_pos += 4;
                sum += mean * block_size;
            }
        }
    }
    T int_mean = sum / (T)size.nbEle;
    return int_mean;
}

inline void extract_block_mean(
    unsigned char *cmpData_pos, int *blocks_mean_quant, size_t num_blocks
){
    unsigned char * qmean_pos = cmpData_pos;
    for(size_t k=0; k<num_blocks; k++){
        blocks_mean_quant[k] = (0xff000000 & (*qmean_pos << 24)) |
                                (0x00ff0000 & (*(qmean_pos+1) << 16)) |
                                (0x0000ff00 & (*(qmean_pos+2) << 8)) |
                                (0x000000ff & *(qmean_pos+3));
        qmean_pos += 4;
    }
}

inline int integerize_quant(
    const int *buffer_pos, int *update_buffer_pos, size_t buffer_dim0_offset, int bias
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_dim0_offset] + buffer_pos[buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    // int res = (center + (sign ? - bias : bias)) >> 2;
    // *update_buffer_pos = res;
    // return res;
    *update_buffer_pos = (center + (sign ? - bias : bias)) >> 2;
    return center;
}

// 1d mean-based
inline int update_block_entries(
    const int *buffer, int *update_buffer, size_t buffer_dim0_offset, int bias, int block_size
){
    int64_t block_sum = 0;
    const int * data_pos = buffer;
    int * update_pos = update_buffer;
    for(int i=0; i<block_size; i++){
        block_sum += integerize_quant(data_pos++, update_pos++, buffer_dim0_offset, bias);
    }
    return static_cast<int>((block_sum >> 2) / block_size);
}

// 2d mean-based
inline int update_block_entries(
    const int *buffer, int *update_buffer, size_t buffer_dim0_offset, int bias, int size_x, int size_y
){
    int64_t block_sum = 0;
    const int * x_data_pos = buffer;
    int * x_update_pos = update_buffer;
    for(int i=0; i<size_x; i++){
        const int * y_data_pos = x_data_pos;
        int * y_update_pos = x_update_pos;
        for(int j=0; j<size_y; j++){
            block_sum += integerize_quant(y_data_pos++, y_update_pos++, buffer_dim0_offset, bias);
        }
        x_data_pos += buffer_dim0_offset;
        x_update_pos += buffer_dim0_offset;
    }
    return static_cast<int>((block_sum >> 2) / (size_x * size_y));
}

template <class Tds>
inline void set_buffer_border_prepred(
    int *integer_buffer, Tds size, int size_x, size_t buffer_dim0_offset,
    TempInfo2D temp_info, bool isTopRow, bool isBottomRow
){
    int i;
    size_t j;
    int * buffer_pos = integer_buffer - buffer_dim0_offset - 1;
    for(i=1; i<size_x+1; i++){
        buffer_pos[i*buffer_dim0_offset] = temp_info.q_W;
        buffer_pos[(i+1)*buffer_dim0_offset-1] = temp_info.q_W;
    }
    if(isTopRow){
        buffer_pos = integer_buffer - buffer_dim0_offset;
        size_t c1 =  size.dim2 * (1.0 - temp_info.ratio) * 0.5 + 1;
        size_t c2 =  size.dim2 * (1.0 + temp_info.ratio) * 0.5 - 1;
        for(j=0; j<size.dim2; j++){
            buffer_pos[j] = temp_info.q_W;
        }
        for(j=c1; j<=c2; j++) buffer_pos[j] = temp_info.q_S;
    }else if(isBottomRow){
        buffer_pos = integer_buffer + size_x * buffer_dim0_offset;
        for(j=0; j<size.dim2; j++) buffer_pos[j] = temp_info.q_W;
    }
}

#endif
