#ifndef _SZP_APP_UTILS_HPP
#define _SZP_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include "SZ_def.hpp"
#include "utils.hpp"

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
    unsigned char **& cmpData_, int **offsets_,
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

struct SZpAppBufferSet_1d
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
    int * decmp_buffer;
    int * cmp_buffer;
    int * rowSum;
    int * lorenzo_buffer;
    int * updateRow_data_pos;
    int * currRow_data_pos;
    int * prevRow_data_pos;
    int * nextRow_data_pos;
    SZpAppBufferSet_1d(
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
                break;
            }
            case appType::HEATDIS:{
                updateBlockRow = buffer_2d + 3 * buffer_size;
                decmp_buffer = buffer_1d;
                cmp_buffer = buffer_1d + buffer_dim0_offset;
                rowSum = buffer_1d + 2 * buffer_dim0_offset;
                lorenzo_buffer = buffer_1d + 3 * buffer_dim0_offset;
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void reset(){
        switch(type){
            case appType::CENTRALDIFF:{
                currRow_data_pos = currBlockRow + 1;
                prevRow_data_pos = prevBlockRow + 1;
                nextRow_data_pos = nextBlockRow + 1;
                break;
            }
            case appType::HEATDIS:{
                memset(buffer_1d, 0, 3 * buffer_dim2 * sizeof(int));
                memset(buffer_2d, 0, 3 * buffer_size * sizeof(int));
                currRow_data_pos = currBlockRow + buffer_dim0_offset + 1;
                prevRow_data_pos = prevBlockRow + buffer_dim0_offset + 1;
                nextRow_data_pos = nextBlockRow + buffer_dim0_offset + 1;
                updateRow_data_pos = updateBlockRow + buffer_dim0_offset + 1;
                break;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
    void reset_residual(){
        residual = 0;
    }
    void prepare_alternative(
    size_t width, size_t x, size_t size_x,
    int *buffer, size_t dim0_offset, int q_w, int *altern){
        int * raw = buffer;
        for(size_t i=0; i<size_x; i++){
            altern[x * width + i] = raw[0] - q_w;
            raw += dim0_offset;
        }
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
    inline void set_cmp_buffer(bool isTopRow){
        if(isTopRow) memset(updateRow_data_pos-buffer_dim0_offset-1, 0, buffer_dim0_offset*sizeof(int));
        else memcpy(updateRow_data_pos-buffer_dim0_offset-1, cmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    inline void save_cmp_buffer_buttom(int Bsize){
        memcpy(cmp_buffer, updateRow_data_pos+(Bsize-1)*buffer_dim0_offset-1, buffer_dim0_offset*sizeof(int));
    }
    inline void prepare_alternative(size_t dim2, const int *raw, int *altern){
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
            case appType::HEATDIS:
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
            case appType::HEATDIS:
            case appType::MEAN:{
                break;
            }
        }
    }
};

template <class T>
inline int predict_lorenzo_1d(
    const T *data_pos, int *buffer_pos, double errorBound
){
    int curr_quant = SZ_quantize(data_pos[0], errorBound);
    buffer_pos[0] = curr_quant;
    int err = curr_quant - buffer_pos[-1];
    return err;
}

inline int predict_lorenzo_1d(
    int *buffer_pos
){
    return buffer_pos[0] - buffer_pos[-1];
}

template <class T>
inline void recover_lorenzo_1d(
    T *data_pos, int *buffer_pos, double errorBound
){
    buffer_pos[0] += buffer_pos[-1];
    data_pos[0] = buffer_pos[0] * 2 * errorBound;
}

inline void recover_lorenzo_1d(
    int *buffer_pos
){
    buffer_pos[0] += buffer_pos[-1];
}

inline void integerize_quant(
    SZpAppBufferSet_1d *buffer_set, const int *buffer_pos, int *update_pos, int bias
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_set->buffer_dim0_offset] + buffer_pos[buffer_set->buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    *update_pos = (center + (sign ? - bias : bias)) >> 2;
}

inline int update_quant_and_predict(
    SZpAppBufferSet_1d *buffer_set, const int *buffer_pos, int *update_pos
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_set->buffer_dim0_offset] + buffer_pos[buffer_set->buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    *update_pos = (center + (sign ? -2 : 2)) >> 2;
    return update_pos[0] - update_pos[-1];
}

inline void integerize_pred_err(
    SZpAppBufferSet_1d *buffer_set, const int *buffer_pos,
    const int *altern, bool flag, int bias, int *update_pos
){
    int center = (flag ? altern[0] : buffer_pos[-1]) + buffer_pos[1] + buffer_pos[-buffer_set->buffer_dim0_offset] + buffer_pos[buffer_set->buffer_dim0_offset];
    int err = center + buffer_set->residual + bias;
    *update_pos = err >> 2;
    buffer_set->residual = (err & 0x3) - bias;
}

inline int update_pred_err_and_predict(
    SZpAppBufferSet_1d *buffer_set, const int *buffer_pos,
    const int *altern, bool flag, int bias, int *update_pos
){
    int center = (flag ? altern[0] : buffer_pos[-1]) + buffer_pos[1] + buffer_pos[-buffer_set->buffer_dim0_offset] + buffer_pos[buffer_set->buffer_dim0_offset];
    int err = center + buffer_set->residual + bias;
    *update_pos = err >> 2;
    buffer_set->residual = (err & 0x3) - bias;
    return update_pos[0];
}

template <class T>
inline int predict_lorenzo_2d(
    const T *data_pos, int *buffer_pos,
    size_t buffer_dim0_offset, double errorBound
){
    int curr_quant = SZ_quantize(data_pos[0], errorBound);
    buffer_pos[0] = curr_quant;
    int err = curr_quant - buffer_pos[-1] - buffer_pos[-buffer_dim0_offset] + buffer_pos[-buffer_dim0_offset-1];
    return err;
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

inline int recover_lorenzo_2d_verb(
    int *buffer_pos, size_t buffer_dim0_offset
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim0_offset] - buffer_pos[-buffer_dim0_offset-1]);
    return buffer_pos[0];
}

inline void integerize_quant(
    const int *buffer_pos, int *update_pos, size_t buffer_dim0_offset, int bias
){
    int center = buffer_pos[-1] + buffer_pos[1] + buffer_pos[-buffer_dim0_offset] + buffer_pos[buffer_dim0_offset];
    unsigned char sign = (center >> 31) & 1;
    *update_pos = (center + (sign ? - bias : bias)) >> 2;
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
    int err = curr_quant - pred;
    return err;
}

inline int predict_lorenzo_3d(
    const int *buffer_pos, size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    int pred = buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
            - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
            - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
    int err = buffer_pos[0] - pred;
    return err;
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

inline int recover_lorenzo_3d_verb(
    int *buffer_pos, size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    buffer_pos[0] += buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset] 
                    - buffer_pos[-buffer_dim1_offset - 1] - buffer_pos[-buffer_dim0_offset - 1] 
                    - buffer_pos[-buffer_dim0_offset - buffer_dim1_offset] + buffer_pos[-buffer_dim0_offset - buffer_dim1_offset - 1];
    return buffer_pos[0];
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
    size_t x, int *integer_buffer, DSize_1d size, size_t size_x,
    SZpAppBufferSet_1d *buffer_set, TempInfo2D& temp_info,
    bool isTopRow, bool isBottomRow
){
    int * buffer_pos = integer_buffer;
    for(int i=0; i<size_x; i++){
        size_t row_ind = x * size.Bwidth + i;
        buffer_pos[-1] = temp_info.q_W + buffer_pos[0];
        buffer_pos[size.dim2] = temp_info.q_W - buffer_set->rowSum[row_ind];
        buffer_pos += buffer_set->buffer_dim0_offset;
    }
    if(isTopRow){
        buffer_pos = integer_buffer - buffer_set->buffer_dim0_offset;
        memcpy(buffer_pos, buffer_set->lorenzo_buffer, size.dim2 * sizeof(int));
    }else if(isBottomRow){
        buffer_pos = integer_buffer + size_x * buffer_set->buffer_dim0_offset;
        for(size_t j=0; j<size.dim2; j++){
            if(!j) buffer_pos[j] = temp_info.q_W;
            else buffer_pos[j] = 0;
        }
    }
}

inline void set_buffer_border_prepred(
    int *integer_buffer, DSize_1d size, size_t buffer_dim0_offset,
    TempInfo2D& temp_info, bool isTopRow, bool isBottomRow
){
    int i;
    size_t j;
    int * buffer_pos = integer_buffer;
    for(i=0; i<size.Bwidth; i++){
        buffer_pos[-1] = temp_info.q_W;
        buffer_pos[size.dim2] = temp_info.q_W;
        buffer_pos += buffer_dim0_offset;
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
        buffer_pos = integer_buffer + size.Bwidth * buffer_dim0_offset;
        for(j=0; j<size.dim2; j++) buffer_pos[j] = temp_info.q_W;
    }
}

inline void set_buffer_border_postpred(
    size_t x, int *integer_buffer, DSize_2d size, int size_x,
    SZpAppBufferSet_2d *buffer_set, TempInfo2D& temp_info,
    bool isTopRow, bool isBottomRow
){
    int i;
    size_t j;
    int * buffer_pos = integer_buffer;
    for(i=0; i<size_x; i++){
        size_t row_ind = x * size.Bsize + i;
        buffer_pos[-1] = buffer_pos[0];
        buffer_pos[size.dim2] = -buffer_set->rowSum[row_ind];
        buffer_pos += buffer_set->buffer_dim0_offset;
    }
    if(isTopRow){
        integer_buffer[-1] += temp_info.q_W;
        integer_buffer[size.dim2] += temp_info.q_W;
        buffer_set->prepare_alternative(size.dim2, integer_buffer, buffer_set->cmp_buffer);
        buffer_pos = integer_buffer - buffer_set->buffer_dim0_offset;
        for(j=0; j<size.dim2; j++){
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
    TempInfo2D& temp_info, bool isTopRow, bool isBottomRow
){
    int i;
    size_t j;
    int * buffer_pos = integer_buffer;
    for(i=0; i<size_x; i++){
        buffer_pos[-1] = temp_info.q_W;
        buffer_pos[size.dim2] = temp_info.q_W;
        buffer_pos += buffer_dim0_offset;
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

// gray-scott
struct gsAppBufferSet
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    size_t buffer_dim1_offset;
    int * buffer_2d;
    int * decmp_buffer;
    int * cmp_buffer;
    int * buffer_3d;
    int * prevBlockPlane;
    int * currBlockPlane;
    int * nextBlockPlane;
    int * updateBlockPlane;
    int * currPlane_data_pos;
    int * prevPlane_data_pos;
    int * nextPlane_data_pos;
    int * updatePlane_data_pos;
    gsAppBufferSet(
    size_t dim1, size_t dim2, size_t dim3, int *buffer_3d_, int *buffer_2d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_),
          buffer_2d(buffer_2d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        buffer_dim0_offset = buffer_dim2 * buffer_dim3;
        buffer_dim1_offset = buffer_dim3;
        decmp_buffer = buffer_2d;
        cmp_buffer = buffer_2d + buffer_dim0_offset;
        prevBlockPlane = buffer_3d;
        currBlockPlane = buffer_3d + buffer_size;
        nextBlockPlane = buffer_3d + 2 * buffer_size;
        updateBlockPlane = buffer_3d + 3 * buffer_size;
    }
    inline void reset(){
        memset(buffer_3d, 0, 3 * buffer_size * sizeof(int));
        memset(buffer_2d, 0, 2 * buffer_dim0_offset * sizeof(int));
        currPlane_data_pos = currBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        prevPlane_data_pos = prevBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        nextPlane_data_pos = nextBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        updatePlane_data_pos = updateBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
    }
    inline void set_decmp_buffer_border(int *buffer_data_pos, size_t size_x){
        int * buffer_top = buffer_data_pos - buffer_dim0_offset - buffer_dim1_offset - 1;
        int * buffer_bottom = buffer_data_pos + size_x * buffer_dim0_offset - buffer_dim1_offset - 1;
        int i, j, k;
        for(i=0; i<=size_x+1; i++){
            int * buffer_pos = buffer_top + i * buffer_dim0_offset;
            int * first_row = buffer_pos, * last_row = buffer_pos + (buffer_dim2 - 1) * buffer_dim1_offset;
            for(k=0; k<buffer_dim3; k++){
                first_row[k] = 0;
                last_row[k] = 0;
            }
            for(j=1; j<=buffer_dim2-2; j++){
                int * row = first_row + j * buffer_dim1_offset;
                row[0] = 0;
                row[buffer_dim3-1] = 0;
            }
        }
    }
    inline void save_decmp_buffer_bottom(int *buffer_data_pos, int Bsize){
        memcpy(decmp_buffer, buffer_data_pos+(Bsize-1)*buffer_dim0_offset-buffer_dim1_offset-1, buffer_dim0_offset*sizeof(int));        
    }
    inline void set_next_decmp_buffer_top(int *buffer_data_pos){
        memcpy(buffer_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, decmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    inline void set_process_buffer(size_t x, bool isTop, bool isBottom, size_t size_x, int border_val){
        int * buffer_top = currPlane_data_pos - buffer_dim0_offset - buffer_dim1_offset - 1;
        int * buffer_bottom = currPlane_data_pos + size_x * buffer_dim0_offset - buffer_dim1_offset - 1;
        if(!isTop){
            const int * prevBlockPlaneBottom_pos = prevPlane_data_pos + (size_x - 1) * buffer_dim0_offset - buffer_dim1_offset - 1;            
            memcpy(buffer_top, prevBlockPlaneBottom_pos, buffer_dim0_offset*sizeof(int));
        }
        if(!isBottom){
            const int * nextBlockPlaneTop_pos = nextPlane_data_pos - buffer_dim1_offset - 1;
            memcpy(buffer_bottom, nextBlockPlaneTop_pos, buffer_dim0_offset*sizeof(int));
        }
        int i, j, k;
        if(isTop) for(i=0; i<buffer_dim0_offset; i++) buffer_top[i] = border_val;
        if(isBottom) for(i=0; i<buffer_dim0_offset; i++) buffer_bottom[i] = border_val;
        for(i=1; i<=size_x+1; i++){
            int * buffer_pos = buffer_top + i * buffer_dim0_offset;
            int * first_row = buffer_pos, * last_row = buffer_pos + (buffer_dim2 - 1) * buffer_dim1_offset;
            for(k=0; k<buffer_dim3; k++){
                first_row[k] = border_val;
                last_row[k] = border_val;
            }
            for(j=1; j<=buffer_dim2-2; j++){
                int * row = first_row + j * buffer_dim1_offset;
                row[0] = border_val;
                row[buffer_dim3-1] = border_val;
            }
        }
    }
    inline void set_cmp_buffer_top(bool isTopRow){
        if(isTopRow) memset(updatePlane_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, 0, buffer_dim0_offset*sizeof(int));
        else memcpy(updatePlane_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, cmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    inline void save_cmp_buffer_buttom(int size_x){
        memcpy(cmp_buffer, updatePlane_data_pos+(size_x-1)*buffer_dim0_offset-buffer_dim1_offset-1, buffer_dim0_offset*sizeof(int));
    }
};

inline void gs_decode_prepred(
    int block_size, int fixed_rate, unsigned char *& encode_pos, SZpCmpBufferSet *cmpkit_set
){
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
}

inline int64_t laplacian(
    const int *buffer_pos, size_t buffer_dim0_offset, size_t buffer_dim1_offset
){
    int64_t res = buffer_pos[-1] + buffer_pos[1] +
            buffer_pos[-buffer_dim1_offset] + buffer_pos[buffer_dim1_offset] +
            buffer_pos[-buffer_dim0_offset] + buffer_pos[buffer_dim0_offset] -
            6 * buffer_pos[0];
    return res;
}

inline int64_t qprod(int64_t v1, int64_t v2, double eb){
    int64_t res = v1 * v2 * eb * 2;
    return res;
}
inline int64_t qprod(int64_t v1, int64_t v2, int64_t v3, double eb){
    int64_t res = v1 * v2 * v3 * eb * eb * 4;
    return res;
}
inline int64_t qprod(int64_t v1, int64_t v2, int64_t v3, int64_t v4, double eb){
    int64_t res = v1 * v2 * v3 * v4 * eb * eb * eb * 8;
    return res;
}

// heatdis3d
struct TempInfo3D
{
	float T_top, T_bott;
    float T_wall, T_init;
    const int q_T, q_B, q_W;
    TempInfo3D(
    float T_t, float T_b, float T_w, float T_i, double eb)
        : T_top(T_t),
          T_bott(T_b),
          T_wall(T_w),
          T_init(T_i),
          q_T(SZ_quantize(T_t, eb)),
          q_B(SZ_quantize(T_b, eb)),
          q_W(SZ_quantize(T_w, eb))
    {}
};

inline void set_buffer_border_prepred(
    int *integer_buffer,
    DSize_3d size, int size_x,
    size_t buffer_dim0_offset,
    size_t buffer_dim1_offset,
    TempInfo3D& temp_info,
    bool isTop,
    bool isBottom
){
    int i, j, k;
    int * buffer_pos = nullptr;
    if(isTop){
        buffer_pos = integer_buffer - buffer_dim0_offset;
        for(j=0; j<size.dim2; j++){
            for(k=0; k<size.dim3; k++){
                buffer_pos[k] = temp_info.q_T;
            }
            buffer_pos += buffer_dim1_offset;
        }
    }else if(isBottom){
        buffer_pos = integer_buffer + size_x * buffer_dim0_offset;
        for(j=0; j<size.dim2; j++){
            for(k=0; k<size.dim3; k++){
                buffer_pos[k] = temp_info.q_B;
            }
            buffer_pos += buffer_dim1_offset;
        }
    }
}

struct ht3DBufferSet
{
    size_t buffer_dim1;
    size_t buffer_dim2;
    size_t buffer_dim3;
    size_t buffer_size;
    size_t buffer_dim0_offset;
    size_t buffer_dim1_offset;
    int * buffer_2d;
    int * decmp_buffer;
    int * cmp_buffer;
    int * buffer_3d;
    int * prevBlockPlane;
    int * currBlockPlane;
    int * nextBlockPlane;
    int * updateBlockPlane;
    int * currPlane_data_pos;
    int * prevPlane_data_pos;
    int * nextPlane_data_pos;
    int * updatePlane_data_pos;
    ht3DBufferSet(
    size_t dim1, size_t dim2, size_t dim3, int *buffer_3d_, int *buffer_2d_)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_dim3(dim3),
          buffer_3d(buffer_3d_),
          buffer_2d(buffer_2d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2 * buffer_dim3;
        buffer_dim0_offset = buffer_dim2 * buffer_dim3;
        buffer_dim1_offset = buffer_dim3;
        decmp_buffer = buffer_2d;
        cmp_buffer = buffer_2d + buffer_dim0_offset;
        prevBlockPlane = buffer_3d;
        currBlockPlane = buffer_3d + buffer_size;
        nextBlockPlane = buffer_3d + 2 * buffer_size;
        updateBlockPlane = buffer_3d + 3 * buffer_size;
    }
    inline void reset(){
        memset(buffer_3d, 0, 3 * buffer_size * sizeof(int));
        memset(buffer_2d, 0, 2 * buffer_dim0_offset * sizeof(int));
        currPlane_data_pos = currBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        prevPlane_data_pos = prevBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        nextPlane_data_pos = nextBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
        updatePlane_data_pos = updateBlockPlane + buffer_dim0_offset + buffer_dim1_offset + 1;
    }
    inline void set_decmp_buffer_border(int *buffer_data_pos, size_t size_x){
        int * buffer_top = buffer_data_pos - buffer_dim0_offset - buffer_dim1_offset - 1;
        int * buffer_bottom = buffer_data_pos + size_x * buffer_dim0_offset - buffer_dim1_offset - 1;
        int i, j, k;
        for(i=0; i<=size_x+1; i++){
            int * buffer_pos = buffer_top + i * buffer_dim0_offset;
            int * first_row = buffer_pos, * last_row = buffer_pos + (buffer_dim2 - 1) * buffer_dim1_offset;
            for(k=0; k<buffer_dim3; k++){
                first_row[k] = 0;
                last_row[k] = 0;
            }
            for(j=1; j<=buffer_dim2-2; j++){
                int * row = first_row + j * buffer_dim1_offset;
                row[0] = 0;
                row[buffer_dim3-1] = 0;
            }
        }
    }
    inline void save_decmp_buffer_bottom(int *buffer_data_pos, int Bsize){
        memcpy(decmp_buffer, buffer_data_pos+(Bsize-1)*buffer_dim0_offset-buffer_dim1_offset-1, buffer_dim0_offset*sizeof(int));        
    }
    inline void set_next_decmp_buffer_top(int *buffer_data_pos){
        memcpy(buffer_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, decmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    inline void set_process_buffer(size_t x, bool isTop, bool isBottom, size_t size_x, int border_val){
        int * buffer_top = currPlane_data_pos - buffer_dim0_offset - buffer_dim1_offset - 1;
        int * buffer_bottom = currPlane_data_pos + size_x * buffer_dim0_offset - buffer_dim1_offset - 1;
        if(!isTop){
            const int * prevBlockPlaneBottom_pos = prevPlane_data_pos + (size_x - 1) * buffer_dim0_offset - buffer_dim1_offset - 1;            
            memcpy(buffer_top, prevBlockPlaneBottom_pos, buffer_dim0_offset*sizeof(int));
        }
        if(!isBottom){
            const int * nextBlockPlaneTop_pos = nextPlane_data_pos - buffer_dim1_offset - 1;
            memcpy(buffer_bottom, nextBlockPlaneTop_pos, buffer_dim0_offset*sizeof(int));
        }
        // int i, j, k;
        // if(isTop) for(i=0; i<buffer_dim0_offset; i++) buffer_top[i] = border_val;
        // if(isBottom) for(i=0; i<buffer_dim0_offset; i++) buffer_bottom[i] = border_val;
        // for(i=1; i<=size_x+1; i++){
        //     int * buffer_pos = buffer_top + i * buffer_dim0_offset;
        //     int * first_row = buffer_pos, * last_row = buffer_pos + (buffer_dim2 - 1) * buffer_dim1_offset;
        //     for(k=0; k<buffer_dim3; k++){
        //         first_row[k] = border_val;
        //         last_row[k] = border_val;
        //     }
        //     for(j=1; j<=buffer_dim2-2; j++){
        //         int * row = first_row + j * buffer_dim1_offset;
        //         row[0] = border_val;
        //         row[buffer_dim3-1] = border_val;
        //     }
        // }
    }
    inline void set_cmp_buffer_top(bool isTopRow){
        if(isTopRow) memset(updatePlane_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, 0, buffer_dim0_offset*sizeof(int));
        else memcpy(updatePlane_data_pos-buffer_dim0_offset-buffer_dim1_offset-1, cmp_buffer, buffer_dim0_offset*sizeof(int));
    }
    inline void save_cmp_buffer_buttom(int size_x){
        memcpy(cmp_buffer, updatePlane_data_pos+(size_x-1)*buffer_dim0_offset-buffer_dim1_offset-1, buffer_dim0_offset*sizeof(int));
    }
};

#endif
