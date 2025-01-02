#ifndef _SZR_APP_UTILS_HPP
#define _SZR_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include "SZ_def.hpp"

template <class T>
inline void compute_regression_coeffcients_2d(
    const T * data_pos, int size_x, int size_y, size_t dim0_offset, float * reg_params_pos
){
	const T * cur_data_pos = data_pos;
	float fx = 0.0;
	float fy = 0.0;
	float f = 0;
	double sum_x; 
	float curData;
	for(size_t i=0; i<size_x; i++){
		sum_x = 0;
		for(size_t j=0; j<size_y; j++){
			curData = *cur_data_pos;
			sum_x += curData;
			fy += curData * j;
			cur_data_pos++;
		}
		fx += sum_x * i;
		f += sum_x;
		cur_data_pos += dim0_offset - size_y;
	}
	float coeff = 1.0 / (size_x * size_y);
	reg_params_pos[0] = (2 * fx / (size_x - 1) - f) * 6 * coeff / (size_x + 1);
	reg_params_pos[1] = (2 * fy / (size_y - 1) - f) * 6 * coeff / (size_y + 1);
	reg_params_pos[2] = f * coeff - ((size_x - 1) * reg_params_pos[0] / 2 + (size_y - 1) * reg_params_pos[1] / 2);
}

template <class T>
inline int predict_regression_2d(
    int i, int j, const float *coeff
){
    T pred = i * coeff[0] + j * coeff[1] + coeff[2];
    return pred;
}

inline void save_regression_coeff_2d(
    unsigned char *& reg_coeff_pos, const float *reg_coeff
){
    for(int i=0; i<3; i++){
        uint32_t tmp;
        memcpy(&tmp, &reg_coeff[i], sizeof(uint32_t));
        for(int k=3; k>=0; k--){
            *(reg_coeff_pos++) = (tmp >> (8 * k)) & 0xff;
        }
    }
}

inline void extract_regression_coeff_2d(
    unsigned char *& reg_coeff_pos, float *reg_coeff
){
    for(int i=0; i<3; i++){
        uint32_t tmp = 0;
        for(int k=3; k>=0; k--){
            tmp |= ((uint32_t)(*reg_coeff_pos++) << (8 * k));
        }
        memcpy(&reg_coeff[i], &tmp, sizeof(uint32_t));
    }
}

inline void extract_regression_coeff_2d(
    unsigned char *& reg_coeff_pos, float *reg_coeff, size_t n
){
    for(size_t i=0; i<n; i++){
        extract_regression_coeff_2d(reg_coeff_pos, reg_coeff + i * 3);
    }
}

template <class T>
inline T compute_prediction_sum(
    int size_x, int size_y, const float *reg_coeff
){
    T pred_sum = size_x * size_y * ((size_x - 1) * reg_coeff[0] * 0.5 + (size_y - 1) * reg_coeff[1] * 0.5 + reg_coeff[2]);
    return pred_sum;
}

template <class T>
inline T compute_prediction_error_sum(
    int block_size, const int *signPredError, double errorBound
){
    T err_sum = 0;
    for(int i=0; i<block_size; i++){
        err_sum += signPredError[i];
    }
    err_sum *= 2 * errorBound;
    return err_sum;
}

struct SZrCmpBufferSet_2d
{
    unsigned char ** cmpData;
    int ** offsets;
    unsigned char * compressed;
    float * reg_coeff;
    unsigned int * absPredError;
    int * signPredError;
    unsigned char * signFlag;
    size_t cmpSize;
    size_t prefix_length;
    SZrCmpBufferSet_2d(
    unsigned char *cmpData_, float *coeff_,
    int *signPredError_, unsigned char *signFlag_)
        : compressed(cmpData_),
          reg_coeff(coeff_),
          signPredError(signPredError_),
          signFlag(signFlag_)
    {}
    SZrCmpBufferSet_2d(
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

struct SZrAppBufferSet_2d
{
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
    SZrAppBufferSet_2d(
    size_t dim1, size_t dim2, int *buffer_2d_, bool update)
        : buffer_dim1(dim1),
          buffer_dim2(dim2),
          buffer_2d(buffer_2d_)
    {
        buffer_size = buffer_dim1 * buffer_dim2;
        buffer_dim0_offset = buffer_dim2;
        prevBlockRow = buffer_2d;
        currBlockRow = buffer_2d + buffer_size;
        nextBlockRow = buffer_2d + 2 * buffer_size;
        if(update) updateBlockRow = buffer_2d + 3 * buffer_size;
    }
    void deriv_reset(){
        currRow_data_pos = currBlockRow;
        prevRow_data_pos = prevBlockRow;
        nextRow_data_pos = nextBlockRow;
    }
    void heatdis_reset(){
        memset(buffer_2d, 0, 3 * buffer_size * sizeof(int));
        updateRow_data_pos = updateBlockRow + buffer_dim0_offset + 1;
        currRow_data_pos = currBlockRow + buffer_dim0_offset + 1;
        prevRow_data_pos = prevBlockRow + buffer_dim0_offset + 1;
        nextRow_data_pos = nextBlockRow + buffer_dim0_offset + 1;
    }
};

#endif
