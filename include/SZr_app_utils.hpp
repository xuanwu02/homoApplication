#ifndef _SZR_APP_UTILS_HPP
#define _SZR_APP_UTILS_HPP

#include <cstdlib>
#include <iostream>
#include "SZ_def.hpp"

template <class T>
inline void compute_regression_coeffcients_2d(
    const T *data_pos, int size_x, int size_y, size_t dim0_offset, float *reg_params_pos
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
inline void 
compute_regression_coeffcients_3d(
    const T *data_pos, int size_x, int size_y, int size_z, size_t dim0_offset, size_t dim1_offset, float *reg_params_pos
){
	const T * cur_data_pos = data_pos;
	float fx = 0.0;
	float fy = 0.0;
	float fz = 0.0;
	float f = 0;
	float sum_x, sum_y; 
	T curData;
	for(int i=0; i<size_x; i++){
		sum_x = 0;
		for(int j=0; j<size_y; j++){
			sum_y = 0;
			for(int k=0; k<size_z; k++){
				curData = *cur_data_pos;
				sum_y += curData;
				fz += curData * k;
				cur_data_pos ++;
			}
			fy += sum_y * j;
			sum_x += sum_y;
			cur_data_pos += (dim1_offset - size_z);
		}
		fx += sum_x * i;
		f += sum_x;
		cur_data_pos += (dim0_offset - size_y * dim1_offset);
	}
	float coeff = 1.0 / (size_x * size_y * size_z);
	reg_params_pos[0] = (2 * fx / (size_x - 1) - f) * 6 * coeff / (size_x + 1);
	reg_params_pos[1] = (2 * fy / (size_y - 1) - f) * 6 * coeff / (size_y + 1);
	reg_params_pos[2] = (2 * fz / (size_z - 1) - f) * 6 * coeff / (size_z + 1);
	reg_params_pos[3] = f * coeff - ((size_x - 1) * reg_params_pos[0] / 2 + (size_y - 1) * reg_params_pos[1] / 2 + (size_z - 1) * reg_params_pos[2] / 2);
}

template <class T>
inline T predict_regression_2d(
    int i, int j, const float *coeff
){
    T pred = i * coeff[0] + j * coeff[1] + coeff[2];
    return pred;
}

template <class T>
inline T predict_regression_3d(
    int i, int j, int k, const float *coeff
){
    T pred = i * coeff[0] + j * coeff[1] + k * coeff[2] + coeff[3];
    return pred;
}

inline void save_regression_coeff_2d(
    unsigned char *& reg_coeff_pos, const float *reg_coeff
){
    for(int i=0; i<REG_COEFF_SIZE_2D; i++){
        uint32_t tmp;
        memcpy(&tmp, &reg_coeff[i], sizeof(uint32_t));
        for(int k=3; k>=0; k--){
            *(reg_coeff_pos++) = (tmp >> (8 * k)) & 0xff;
        }
    }
}

inline void save_regression_coeff_3d(
    unsigned char *& reg_coeff_pos, const float *reg_coeff
){
    for(int i=0; i<REG_COEFF_SIZE_3D; i++){
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
    for(int i=0; i<REG_COEFF_SIZE_2D; i++){
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
        extract_regression_coeff_2d(reg_coeff_pos, reg_coeff + i * REG_COEFF_SIZE_2D);
    }
}

inline void extract_regression_coeff_3d(
    unsigned char *& reg_coeff_pos, float *reg_coeff
){
    for(int i=0; i<REG_COEFF_SIZE_3D; i++){
        uint32_t tmp = 0;
        for(int k=3; k>=0; k--){
            tmp |= ((uint32_t)(*reg_coeff_pos++) << (8 * k));
        }
        memcpy(&reg_coeff[i], &tmp, sizeof(uint32_t));
    }
}

inline void extract_regression_coeff_3d(
    unsigned char *& reg_coeff_pos, float *reg_coeff, size_t n
){
    for(size_t i=0; i<n; i++){
        extract_regression_coeff_2d(reg_coeff_pos, reg_coeff + i * REG_COEFF_SIZE_3D);
    }
}

inline double compute_prediction_sum_2d(
    int size_x, int size_y, int block_size, float *reg_coeff
){
    double pred_sum = block_size * ((size_x - 1) * reg_coeff[0] * 0.5 +
                                    (size_y - 1) * reg_coeff[1] * 0.5 +
                                    reg_coeff[2]);
    return pred_sum;
}

inline double compute_prediction_sum_3d(
    int size_x, int size_y, int size_z, int block_size, float *reg_coeff
){
    double pred_sum = block_size * ((size_x - 1) * reg_coeff[0] * 0.5 +
                                    (size_y - 1) * reg_coeff[1] * 0.5 +
                                    (size_z - 1) * reg_coeff[2] * 0.5 +
                                    reg_coeff[3]);
    return pred_sum;
}

inline double compute_prediction_error_sum(
    int block_size, const int *signPredError, double errorBound
){
    int64_t int_err_sum = 0;
    for(int i=0; i<block_size; i++){
        int_err_sum += signPredError[i];
    }
    double err_sum = int_err_sum * 2 * errorBound;
    return err_sum;
}

inline double compute_mean_2d(
    DSize_2d& size, unsigned char *cmpData, unsigned char *signFlag,
    int *signPredError, unsigned char *reg_coeff_pos, float *reg_coeff, double errorBound
){
    double mean = 0;
    int block_ind = 0;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_2D * FLOAT_BYTES) * size.num_blocks;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            extract_regression_coeff_2d(reg_coeff_pos, reg_coeff);
            mean += compute_prediction_sum_2d(size_x, size_y, block_size, reg_coeff);
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                mean += compute_prediction_error_sum(block_size, signPredError, errorBound);
            }
        }
    }
    mean /= size.nbEle;
    return mean;
}

inline double compute_mean_3d(
    DSize_3d& size, unsigned char *cmpData, unsigned char *signFlag,
    int *signPredError, unsigned char *reg_coeff_pos, float *reg_coeff, double errorBound
){
    double mean = 0;
    int block_ind = 0;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + REG_COEFF_SIZE_3D * FLOAT_BYTES) * size.num_blocks;
    for(size_t x=0; x<size.block_dim1; x++){
        int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            for(size_t z=0; z<size.block_dim3; z++){
                int size_z = ((z+1)*size.Bsize < size.dim3) ? size.Bsize : size.dim3 - z*size.Bsize;
                int block_size = size_x * size_y * size_z;
                int fixed_rate = (int)cmpData[block_ind++];
                extract_regression_coeff_3d(reg_coeff_pos, reg_coeff);
                mean += compute_prediction_sum_3d(size_x, size_y, size_z, block_size, reg_coeff);
                if(fixed_rate){
                    size_t cmp_block_sign_length = (block_size + 7) / 8;
                    convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                    encode_pos += cmp_block_sign_length;
                    unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                    encode_pos += savedbitsbytelength;
                    convert2SignIntArray(signFlag, signPredError, block_size);
                    mean += compute_prediction_error_sum(block_size, signPredError, errorBound);
                }
            }
        }
    }
    mean /= size.nbEle;
    return mean;
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
    SZrAppBufferSet_2d(
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
                currRow_data_pos = currBlockRow;
                prevRow_data_pos = prevBlockRow;
                nextRow_data_pos = nextBlockRow;
            }
            case appType::MEAN:{
                break;
            }
        }
    }
};

#endif
