#ifndef _SZP_APP_UTILS_HPP
#define _SZP_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include "SZ_def.hpp"

struct Temperature_info
{
    float ratio;
    const int q_S;
    const int q_W;
    const int q_X;
    Temperature_info(
    float source_temp, float wall_temp, float ratio, double eb)
        : ratio(ratio), 
          q_S(SZ_quantize(source_temp, eb)),
          q_W(SZ_quantize(wall_temp, eb)),
          q_X(q_S - q_W)
    {}
    void prepare_src_row(size_t dim2, int *quant_buffer, int *pred_err_buffer){
        size_t c1 =  dim2 * (1.0 - ratio) * 0.5 + 1;
        size_t c2 =  dim2 * (1.0 + ratio) * 0.5 - 1;
        size_t j;
        for(j=0; j<dim2; j++){
            quant_buffer[j] = q_S;
            if(j < c1 || j > c2) quant_buffer[j] = q_W;
        }
        int prev_quant = 0;
        for(j=0; j<dim2; j++){
            pred_err_buffer[j] = quant_buffer[j] - prev_quant;
            prev_quant = quant_buffer[j];
        }
    }
};

struct SZpCmpBufferSet
{
    unsigned char ** cmpData;
    int ** offsets;
    unsigned char * compressed;
    unsigned int * absPredError;
    int * signPredError;
    unsigned char * signFlag;
    size_t cmpSize;
    size_t prefix_length;
    SZpCmpBufferSet(
    unsigned char *cmpData_,
    int *signPredError_, unsigned char *signFlag_)
        : compressed(cmpData_),
          signPredError(signPredError_),
          signFlag(signFlag_)
    {}
    SZpCmpBufferSet(
    unsigned char **cmpData_, int **offsets_,
    unsigned int *absPredError_, int *signPredError_, unsigned char *signFlag_)
        : cmpData(cmpData_),
          offsets(offsets_),
          absPredError(absPredError_),
          signPredError(signPredError_),
          signFlag(signFlag_)
    {}
    void reset(){
        cmpSize = 0;
        prefix_length = 0;
    }
};

struct SZpAppBufferSet_2d
{
    appType type;
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    int residual;
    int * buffer_2d;
    int * prevBlockRow;
    int * currBlockRow;
    int * nextBlockRow;
    int * updateBlockRow;
    int * buffer_1d;
    int * dy_buffer;
    int * decmp_buffer;
    int * cmp_buffer;
    int * rowSum;
    int * colSum;
    int * lorenzo_buffer;
    int * updateRow_data_pos;
    int * currRow_data_pos;
    int * prevRow_data_pos;
    int * nextRow_data_pos;
    SZpAppBufferSet_2d(
    size_t dim1, size_t dim2, int *buffer_2d_, int *buffer_1d_, appType type_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_2d(buffer_2d_),
          buffer_1d(buffer_1d_),
          type(type_)
    {
        buffer_size = buffer_dim1 * buffer_dim2;
        buffer_dim0_offset = buffer_dim2;
        prevBlockRow = buffer_2d;
        currBlockRow = buffer_2d + buffer_size;
        nextBlockRow = buffer_2d + 2 * buffer_size;
        switch(type){
            case appType::CENTRALDIFF:{
                decmp_buffer = buffer_1d;
                dy_buffer = buffer_1d + buffer_dim0_offset;
                break;
            }
            case appType::HEATDIS:{
                updateBlockRow = buffer_2d + 3 * buffer_size;
                decmp_buffer = buffer_1d;
                cmp_buffer = buffer_1d + buffer_dim0_offset;
                rowSum = buffer_1d + 2 * buffer_dim0_offset;
                colSum = buffer_1d + 3 * buffer_dim0_offset;
                lorenzo_buffer = buffer_1d + 4 * buffer_dim0_offset;
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void reset(){
        currRow_data_pos = currBlockRow + buffer_dim0_offset + 1;
        prevRow_data_pos = prevBlockRow + buffer_dim0_offset + 1;
        nextRow_data_pos = nextBlockRow + buffer_dim0_offset + 1;
        switch(type){
            case appType::CENTRALDIFF:{
                memset(buffer_1d, 0, 2 * buffer_dim2 * sizeof(int));
                break;
            }
            case appType::HEATDIS:{
                residual = 0;
                memset(buffer_1d, 0, 4 * buffer_dim2 * sizeof(int));
                memset(buffer_2d, 0, 3 * buffer_size * sizeof(int));
                updateRow_data_pos = updateBlockRow + buffer_dim0_offset + 1;
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void set_cmp_buffer(bool isTopRow){
        if(isTopRow) memset(updateRow_data_pos-buffer_dim0_offset-1, 0, buffer_dim0_offset*sizeof(int));
        else memcpy(updateRow_data_pos-buffer_dim0_offset-1, cmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    void copy_buffer_buttom(int Bsize){
        memcpy(cmp_buffer, updateRow_data_pos+(Bsize-1)*buffer_dim0_offset-1, buffer_dim0_offset*sizeof(int));
    }
    void prepare_alternative(size_t dim2, const int *raw, int *altern){
        for(size_t j=0; j<dim2; j++){
            altern[j] = raw[j] - lorenzo_buffer[j];
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

struct SZpAppBufferSet_3d
{
    appType type;
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    size_t buffer_dim1_offset;
    int * buffer_3d;
    int * buffer_2d;
    int * prevBlockPlane;
    int * currBlockPlane;
    int * nextBlockPlane;
    int * decmp_buffer;
    int * currPlane_data_pos;
    int * prevPlane_data_pos;
    int * nextPlane_data_pos;
    SZpAppBufferSet_3d(
    size_t dim1, size_t dim2, size_t dim3,
    int *buffer_3d_, int *buffer_2d_, appType type_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_),
          buffer_2d(buffer_2d_),
          type(type_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        buffer_dim0_offset = buffer_dim2 * buffer_dim3;
        buffer_dim1_offset = buffer_dim3;
        prevBlockPlane = buffer_3d;
        currBlockPlane = buffer_3d + buffer_size;
        nextBlockPlane = buffer_3d + 2 * buffer_size;
        switch(type){
            case appType::CENTRALDIFF:{
                decmp_buffer = buffer_2d;
                break;
            }
            case appType::HEATDIS:{
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void reset(){
        currPlane_data_pos = currBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        prevPlane_data_pos = prevBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        nextPlane_data_pos = nextBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        switch(type){
            case appType::CENTRALDIFF:{
                memset(buffer_2d, 0, buffer_dim0_offset * sizeof(int));
                break;
            }
            case appType::HEATDIS:{
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
};


template <class T>
inline int predict_lorenzo_2d(
    const T *data_pos, int *buffer_pos,
    size_t buffer_dim0_offset, double errorBound
){
    int curr_quant = SZ_quantize(data_pos[0], errorBound);
    buffer_pos[0] = curr_quant;
    int pred = curr_quant - buffer_pos[-1] - buffer_pos[-buffer_dim0_offset] + buffer_pos[-buffer_dim0_offset-1];
    return pred;
}

inline int predict_lorenzo_2d(
    const int *buffer_pos, size_t buffer_dim0_offset
){
    return buffer_pos[0] - buffer_pos[-1] - buffer_pos[-buffer_dim0_offset] + buffer_pos[-buffer_dim0_offset-1];
}

template <class T>
inline void recover_lorenzo_2d(
    T *data_pos, int *buffer_pos,
    size_t buffer_dim0_offset, double errorBound
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim0_offset] - buffer_pos[-buffer_dim0_offset-1]);
    data_pos[0] = buffer_pos[0] * 2 * errorBound;
}

template <class T>
inline void recover_lorenzo_2d(
    T& quant_sum, int *buffer_pos, size_t buffer_dim0_offset
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim0_offset] - buffer_pos[-buffer_dim0_offset-1]);
    quant_sum += buffer_pos[0];
}

inline void recover_lorenzo_2d(
    int *buffer_pos, size_t buffer_dim0_offset
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim0_offset] - buffer_pos[-buffer_dim0_offset-1]);
}

inline void integerize_quant(
    const int *buffer_pos, int *update_pos, size_t buffer_dim0_offset
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_dim0_offset] + buffer_pos[buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    *update_pos = (center + (sign ? -2 : 2)) >> 2;
}

inline int update_quant_and_predict(
    SZpAppBufferSet_2d *buffer_set, const int *buffer_pos, int *update_pos
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_set->buffer_dim0_offset] + buffer_pos[buffer_set->buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    *update_pos = (center + (sign ? -2 : 2)) >> 2;
    return update_pos[0] - update_pos[-1] - update_pos[-buffer_set->buffer_dim0_offset] + update_pos[-buffer_set->buffer_dim0_offset-1];
}

inline void integerize_pred_err(
    SZpAppBufferSet_2d *buffer_set, const int *buffer_pos,
    const int *altern, bool flag, int bias, int *update_pos
){
    int center = buffer_pos[-1] + buffer_pos[1] + (flag ? altern[0] : buffer_pos[-buffer_set->buffer_dim0_offset]) + buffer_pos[buffer_set->buffer_dim0_offset];
    int err = center + buffer_set->residual + bias;
    *update_pos = err >> 2;
    buffer_set->residual = (err & 0x3) - bias;
}

inline int update_pred_err_and_predict(
    SZpAppBufferSet_2d *buffer_set, const int *buffer_pos,
    const int *altern, bool flag, int bias, int *update_pos
){
    int center = buffer_pos[-1] + buffer_pos[1] + (flag ? altern[0] : buffer_pos[-buffer_set->buffer_dim0_offset]) + buffer_pos[buffer_set->buffer_dim0_offset];
    int err = center + buffer_set->residual + bias;
    *update_pos = err >> 2;
    buffer_set->residual = (err & 0x3) - bias;
    return update_pos[0];
}

template <class T>
inline int predict_lorenzo_3d(
    const T *data_pos, int *buffer_pos, double errorBound,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    int curr_quant = SZ_quantize(data_pos[0], errorBound);
    buffer_pos[0] = curr_quant;
    int pred = buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
            - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
            - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
    int diff = curr_quant - pred;
    return diff;
}

template <class T>
inline void recover_lorenzo_3d(
    T *data_pos, int *buffer_pos, double errorBound,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
                    - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
                    - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
    data_pos[0] = buffer_pos[0] * 2 * errorBound;
}

template <class T>
inline void recover_lorenzo_3d(
    T& quant_sum, int *buffer_pos,
    size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
                    - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
                    - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
    quant_sum += buffer_pos[0];
}

inline void recover_lorenzo_3d(
    int *buffer_pos, size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
                    - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
                    - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
}

template <class T>
inline void deriv_lorenzo_2d(
    const int *pred_level_0_pos, const int *pred_level_1_pos,
    int *res_int_buffer_pos, T *res_pos, size_t res_buffer_dim1_offset, double errorBound
){
    int res_integer = pred_level_0_pos[0] + pred_level_1_pos[0]
                    + res_int_buffer_pos[-1] + res_int_buffer_pos[-res_buffer_dim1_offset] - res_int_buffer_pos[-res_buffer_dim1_offset-1];
    res_int_buffer_pos[0] = res_integer;
    res_pos[0] = res_integer * errorBound;
}

inline void set_buffer_border_postpred(
    size_t x, int *integer_buffer, DSize_2d size, int size_x,
    SZpAppBufferSet_2d *buffer_set, Temperature_info temp_info,
    bool isTopRow, bool isBottomRow
){
    int i;
    size_t j;
    size_t row_ind, col_ind;
    int * buffer_pos = integer_buffer - buffer_set->buffer_dim0_offset - 1;
    for(i=0; i<size_x; i++){
        row_ind = x * size.Bsize + i;
        buffer_pos[(i+1)*buffer_set->buffer_dim0_offset] = buffer_pos[(i+1)*buffer_set->buffer_dim0_offset+1];
        buffer_pos[(i+2)*buffer_set->buffer_dim0_offset-1] = -buffer_set->rowSum[row_ind];
    }
    if(isTopRow){
        integer_buffer[-1] += temp_info.q_W;
        integer_buffer[size.dim2] += temp_info.q_W;
        buffer_set->prepare_alternative(size.dim2, integer_buffer, buffer_set->cmp_buffer);
        buffer_pos = integer_buffer - buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            buffer_pos[j] = integer_buffer[j] + buffer_set->lorenzo_buffer[j];
        }
        buffer_pos[1] -= temp_info.q_W;
    }else if(isBottomRow){
        buffer_pos = integer_buffer + size_x * buffer_set->buffer_dim0_offset;
        for(j=0; j<size.dim2; j++) buffer_pos[j] = -buffer_set->colSum[j];
        buffer_pos[0] += temp_info.q_W;
    }
}

inline void set_buffer_border_prepred(
    int *integer_buffer, DSize_2d size, int size_x, size_t buffer_dim0_offset,
    Temperature_info temp_info, bool isTopRow, bool isBottomRow
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
