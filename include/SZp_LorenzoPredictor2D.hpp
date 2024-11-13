#ifndef _SZP_HEATDIS_2DLORENZO_HPP
#define _SZP_HEATDIS_2DLORENZO_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include "typemanager.hpp"
#include "application_utils.hpp"

struct DSize_2d
{
	size_t dim1;
	size_t dim2;
	size_t nbEle;
	int Bsize;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t num_blocks;
	size_t dim1_offset;
	DSize_2d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		max_num_block_elements = bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2;
		dim1_offset = r2;
	}
};

template <class T>
inline int predict_lorenzo_2d(
    const T *data_pos, int *buffer_pos,
    size_t buffer_dim1_offset, double errorBound
){
    int curr_quant = SZp_quantize(data_pos[0], errorBound);
    buffer_pos[0] = curr_quant;
    int diff = curr_quant - buffer_pos[-1] - buffer_pos[-buffer_dim1_offset] + buffer_pos[-buffer_dim1_offset-1];
    return diff;
}

template <class T>
void SZp_compress_2dLorenzo(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    unsigned int * absQuantDiff = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    const T * x_data_pos = oriData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_buffer + buffer_dim1_offset + 1; // (1,1)
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            unsigned int * abs_diff_pos = absQuantDiff;
            unsigned char * sign_pos = signFlag;
            int quant_diff, max_quant_diff = 0;
            int * block_buffer_pos = buffer_start_pos;
            const T * curr_data_pos = y_data_pos;
            for(int i=0; i<size_x; i++){
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    quant_diff = predict_lorenzo_2d(curr_data_pos, curr_buffer_pos, buffer_dim1_offset, errorBound);
                    curr_data_pos++;
                    curr_buffer_pos++;
                    (*sign_pos++) = (quant_diff < 0);
                    unsigned int abs_diff = abs(quant_diff);
                    (*abs_diff_pos++) = abs_diff;
                    max_quant_diff = max_quant_diff > abs_diff ? max_quant_diff : abs_diff;
                }
                block_buffer_pos += buffer_dim1_offset; // shift down by one data row
                curr_data_pos += size.dim1_offset - size_y; // right most of curr data block row -> left most of next data block row
            }
            buffer_start_pos += size.Bsize; // shift right by one block
            y_data_pos += size.Bsize; // shift right by one block
            int fixed_rate = max_quant_diff == 0 ? 0 : INT_BITS - __builtin_clz(max_quant_diff);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, cmpData_pos);
                cmpData_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absQuantDiff, block_size, cmpData_pos, fixed_rate);
                cmpData_pos += savedbitsbyteLength;
            }
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim1_offset, buffer_dim1_offset*sizeof(int));
        x_data_pos += size.Bsize * size.dim2; // shift down by one row of data blocks
    }
    cmpSize = cmpData_pos - cmpData;
    free(quant_buffer);
    free(absQuantDiff);
    free(signFlag);
}

inline void conver2SignIntArray(
    const unsigned char *signFlag,
    int *signQuantDiff, int n
){
    for(int i=0; i<n; i++){
        if(signFlag[i]) signQuantDiff[i] *= -1;
    }
}

template <class T>
inline void recover_lorenzo_2d(
    T *data_pos, int *buffer_pos,
    size_t buffer_dim1_offset, double errorBound
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] - buffer_pos[-buffer_dim1_offset-1]);
    data_pos[0] = buffer_pos[0] * 2 * errorBound;
}

template <class T>
inline void recover_lorenzo_2d(
    T& quant_sum, int *buffer_pos,
    size_t buffer_dim1_offset, double errorBound
){
    buffer_pos[0] += (buffer_pos[-1] + buffer_pos[-buffer_dim1_offset] - buffer_pos[-buffer_dim1_offset-1]);
    quant_sum += buffer_pos[0];
}

template <class T>
void SZp_decompress_2dLorenzo(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1_offset = size.dim2 + 1;
    int * quant_diff_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_diff_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signQuantDiff = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    T * x_data_pos = decData;
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        int * buffer_start_pos = quant_diff_buffer + buffer_dim1_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            T * curr_data_pos = y_data_pos;
            int quant_sum = 0;
            if(!fixed_rate){
                memset(signQuantDiff, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signQuantDiff, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                conver2SignIntArray(signFlag, signQuantDiff, block_size);
            }
            for(int i=0; i<size_x; i++){
                memcpy(block_buffer_pos, signQuantDiff+i*size_y, size_y*sizeof(int));
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    recover_lorenzo_2d(curr_data_pos, curr_buffer_pos, buffer_dim1_offset, errorBound);
                    curr_data_pos++;
                    curr_buffer_pos++;
                }
                block_buffer_pos += buffer_dim1_offset; // shift down by one data row
                curr_data_pos += size.dim1_offset - size_y; // right most of curr data block row -> left most of next data block row
            }
            buffer_start_pos += size.Bsize; // shift right by one block
            y_data_pos += size.Bsize; // shift right by one block
        }
        memcpy(quant_diff_buffer, quant_diff_buffer+size.Bsize*buffer_dim1_offset, buffer_dim1_offset*sizeof(int));
        x_data_pos += size.Bsize * size.dim2; // shift down by one row of data blocks
    }
    free(quant_diff_buffer);
    free(signQuantDiff);
    free(signFlag);
}

/**
 * @brief Compressed data recovered to post-prediciton state
 * @return Mean value
*/
template <class T>
T SZp_mean_2dLorenzo_recover2PostPrediction(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    int * signQuantDiff = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int index_x = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int index_y = 0;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signQuantDiff, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                conver2SignIntArray(signFlag, signQuantDiff, block_size);
                const int * diff_pos = signQuantDiff;
                for(int i=0; i<size_x; i++){
                    for(int j=0; j<size_y; j++){
                        quant_sum += (size.dim1 - (index_x + i)) * (size.dim2 - (index_y + j)) * (*diff_pos++);
                    }
                }
            }
            index_y += size.Bsize;
        }
        index_x += size.Bsize;
    }
    free(signQuantDiff);
    free(signFlag);
    T mean = (quant_sum / size.nbEle) * 2 * errorBound;
    return mean;
}

/**
 * @brief Compressed data recovered to pre-prediciton state
 * @return Mean value
*/
template <class T>
T SZp_mean_2dLorenzo_recover2PrePrediction(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t buffer_dim1_offset = size.dim2 + 1;
    int * quant_buffer = (int *)malloc((size.Bsize+1)*(size.dim2+1)*sizeof(int));
    memset(quant_buffer, 0, (size.Bsize+1)*(size.dim2+1)*sizeof(int));
    int * signQuantDiff = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    int block_ind = 0;
    int64_t quant_sum = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        int * buffer_start_pos = quant_buffer + buffer_dim1_offset + 1;
        for(size_t y=0; y<size.block_dim2; y++){
            int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
            int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int block_size = size_x * size_y;
            int fixed_rate = (int)cmpData[block_ind++];
            int * block_buffer_pos = buffer_start_pos;
            if(!fixed_rate){
                memset(signQuantDiff, 0, size.max_num_block_elements*sizeof(int));
            }else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
                cmpData_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signQuantDiff, fixed_rate);
                cmpData_pos += savedbitsbytelength;
                conver2SignIntArray(signFlag, signQuantDiff, block_size);
            }
            for(int i=0; i<size_x; i++){
                memcpy(block_buffer_pos, signQuantDiff+i*size_y, size_y*sizeof(int));
                int * curr_buffer_pos = block_buffer_pos;
                for(int j=0; j<size_y; j++){
                    recover_lorenzo_2d(quant_sum, curr_buffer_pos, buffer_dim1_offset, errorBound);
                    curr_buffer_pos++;
                }
                block_buffer_pos += buffer_dim1_offset; // shift down by one data row
            }
            buffer_start_pos += size.Bsize; // shift right by one block
        }
        memcpy(quant_buffer, quant_buffer+size.Bsize*buffer_dim1_offset, buffer_dim1_offset*sizeof(int));
    }
    free(quant_buffer);
    free(signQuantDiff);
    free(signFlag);
    T mean = (quant_sum / size.nbEle) * 2 * errorBound;
    return mean;
}

/**
 *@param x Block row index
 *@brief Recover one full block row to post-prediciton state.
 No padding for buffer
*/
inline void recoverBlockRow2PostPrediction(
    size_t x, const unsigned char *cmpData,
    unsigned char *& cmpData_pos, DSize_2d size, int *buffer_pos,
    unsigned char *signFlag, int *signQuantDiff
){
    int block_ind = x * size.block_dim2;
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    int * buffer_start_pos = buffer_pos;
    for(size_t y=0; y<size.block_dim2; y++){
        int size_y = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
        int block_size = size_x * size_y;
        int * curr_buffer_pos = buffer_start_pos;
        int fixed_rate = (int)cmpData[block_ind++];
        if(!fixed_rate){
            for(int i=0; i<size_x; i++){
                memset(curr_buffer_pos, 0, size_y*sizeof(int));
                curr_buffer_pos += size.dim2;
            }
        }
        else{
            size_t cmp_block_sign_length = (block_size + 7) / 8;
            convertByteArray2IntArray_fast_1b_args(block_size, cmpData_pos, cmp_block_sign_length, signFlag);
            cmpData_pos += cmp_block_sign_length;
            unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(cmpData_pos, block_size, signQuantDiff, fixed_rate);
            cmpData_pos += savedbitsbytelength;
            conver2SignIntArray(signFlag, signQuantDiff, block_size);
            int * data_pos = signQuantDiff;
            for(int i=0; i<size_x; i++){
                memcpy(curr_buffer_pos, data_pos, size_y*sizeof(int));
                curr_buffer_pos += size.dim2;
                data_pos += size_y;
            }
        }        
        buffer_start_pos += size.Bsize;
    }
}

template <class T>
inline void dxdyProcessTopBlockRow(
    size_t x, DSize_2d size, T *row_buffer, T *col_buffer,
    const int *currBlockRow, const int *nextBlockRow, double errorBound,
    unsigned char *signFlag, int *signQuantDiff,
    T *& dx_pos, T *& dy_pos
){
    memset(col_buffer, 0, size.Bsize*sizeof(T));
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * dx_buffer_pos = currBlockRow;
    const int * dy_buffer_pos = currBlockRow;
    const int * next_buffer_pos = nextBlockRow;
    int i, j;
    {
        i = 0;
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += *(dy_buffer_pos+size.dim2) * 2 * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += *(dy_buffer_pos+size.dim2) * 2 * errorBound;
            *dy_pos++ = col_buffer[i];
            dy_buffer_pos++;
        }
        // rightmost
        j = size.dim2 - 1;
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += *(dy_buffer_pos+size.dim2) * 2 * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
    }
    for(i=1; i<size_x-1; i++){
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
            *dy_pos++ = col_buffer[i];
            dy_buffer_pos++;
        }
        // rightmost
        j = size.dim2 - 1;
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
    }
    {   
        i = size_x - 1;
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
        *dy_pos++ = col_buffer[i];
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
            *dy_pos++ = col_buffer[i];
        }
        // rightmost
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
        *dy_pos++ = col_buffer[i];
    }
}

template <class T>
inline void dxdyProcessCentralBlockRow(
    size_t x, DSize_2d size, T *row_buffer, T *col_buffer,
    const int *currBlockRow, const int *nextBlockRow, double errorBound,
    unsigned char *signFlag, int *signQuantDiff,
    T *& dx_pos, T *& dy_pos
){
    memset(col_buffer, 0, size.Bsize*sizeof(T));
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * dx_buffer_pos = currBlockRow;
    const int * dy_buffer_pos = currBlockRow;
    const int * next_buffer_pos = nextBlockRow;
    int i, j;
    for(i=0; i<size_x-1; i++){
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
            *dy_pos++ = col_buffer[i];
            dy_buffer_pos++;
        }
        // rightmost
        j = size.dim2 - 1;
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
    }
    {   
        i = size_x - 1;
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
        *dy_pos++ = col_buffer[i];
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
            *dy_pos++ = col_buffer[i];
        }
        // rightmost
        j = size.dim2 - 1;
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos++ + *next_buffer_pos++) * errorBound;
        *dy_pos++ = col_buffer[i];
    }
}

template <class T>
inline void dxdyProcessBottomBlockRow(
    size_t x, DSize_2d size, T *row_buffer, T *col_buffer,
    const int *currBlockRow, double errorBound,
    unsigned char *signFlag, int *signQuantDiff,
    T *& dx_pos, T *& dy_pos
){
    memset(col_buffer, 0, size.Bsize*sizeof(T));
    int size_x = ((x+1)*size.Bsize < size.dim1) ? size.Bsize : size.dim1 - x*size.Bsize;
    const int * dx_buffer_pos = currBlockRow;
    const int * dy_buffer_pos = currBlockRow;
    int i, j;
    for(i=0; i<size_x-1; i++){
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
            *dy_pos++ = col_buffer[i];
            dy_buffer_pos++;
        }
        // rightmost
        j = size.dim2 - 1;
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += (*dy_buffer_pos + *(dy_buffer_pos+size.dim2)) * errorBound;
        *dy_pos++ = col_buffer[i];
        dy_buffer_pos++;
    }
    {   // i = size_x - 1;
        // leftmost
        j = 0;
        row_buffer[j] += *(dx_buffer_pos+1) * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += *dy_buffer_pos++ * 2 * errorBound;
        *dy_pos++ = col_buffer[i];
        // central
        for(j=1; j<size.dim2-1; j++){
            row_buffer[j] += (*dx_buffer_pos + *(dx_buffer_pos+1)) * errorBound;
            *dx_pos++ = row_buffer[j];
            dx_buffer_pos++;
            col_buffer[i] += *dy_buffer_pos++ * 2 * errorBound;
            *dy_pos++ = col_buffer[i];
        }
        // rightmost
        row_buffer[j] += *dx_buffer_pos * 2 * errorBound;
        *dx_pos++ = row_buffer[j];
        dx_buffer_pos++;
        col_buffer[i] += *dy_buffer_pos++ * 2 * errorBound;
        *dy_pos++ = col_buffer[i];
    }
}

/**
 * @brief Compressed data recovered to post-prediciton state
 * @return Central difference
*/
template <class T>
void SZp_dxdy_2dLorenzo_recover2PostPrediction(
    unsigned char *cmpData, size_t dim1, size_t dim2,
    int blockSideLength, double errorBound,
    T *dx_result, T *dy_result
){
    const DSize_2d size(dim1, dim2, blockSideLength);
    size_t res_dim1_offset = size.dim2 + 1;
    int buffer_dim1_offset = size.Bsize + 1;
    int * currBlockRow = (int *)malloc(size.Bsize*size.dim2*sizeof(int));
    memset(currBlockRow, 0, size.Bsize*size.dim2*sizeof(int));
    int * nextBlockRow = (int *)malloc(size.Bsize*size.dim2*sizeof(int));
    memset(nextBlockRow, 0, size.Bsize*size.dim2*sizeof(int));
    T * row_buffer = (T *)malloc(size.dim2*sizeof(T));
    memset(row_buffer, 0, size.dim2*sizeof(T));
    T * col_buffer = (T *)malloc(size.Bsize*sizeof(T));
    memset(col_buffer, 0, size.Bsize*sizeof(T));
    int * signQuantDiff = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    unsigned char * cmpData_pos = cmpData + size.num_blocks;
    memset(dx_result, 0, size.dim1*size.dim2*sizeof(T));
    memset(dy_result, 0, size.dim1*size.dim2*sizeof(T));
    int * currBlockRow_pos = currBlockRow;
    int * nextBlockRow_pos = nextBlockRow;
    T * dx_pos = dx_result;
    T * dy_pos = dy_result;
    int * tempBlockRow = nullptr;
    size_t x;
    x = 0;
    recoverBlockRow2PostPrediction(x, cmpData, cmpData_pos, size, currBlockRow_pos, signFlag, signQuantDiff);
    recoverBlockRow2PostPrediction(x+1, cmpData, cmpData_pos, size, nextBlockRow_pos, signFlag, signQuantDiff);
    dxdyProcessTopBlockRow(x, size, row_buffer, col_buffer, currBlockRow_pos, nextBlockRow_pos, errorBound, signFlag, signQuantDiff, dx_pos, dy_pos);
    for(x=1; x<size.block_dim1-1; x++){
        tempBlockRow = currBlockRow_pos;
        currBlockRow_pos = nextBlockRow_pos;
        nextBlockRow_pos = tempBlockRow;
        recoverBlockRow2PostPrediction(x+1, cmpData, cmpData_pos, size, nextBlockRow_pos, signFlag, signQuantDiff);
        dxdyProcessCentralBlockRow(x, size, row_buffer, col_buffer, currBlockRow_pos, nextBlockRow_pos, errorBound, signFlag, signQuantDiff, dx_pos, dy_pos);
    }
    x = size.block_dim1 - 1;
    currBlockRow_pos = nextBlockRow_pos;
    dxdyProcessBottomBlockRow(x, size, row_buffer, col_buffer, currBlockRow_pos, errorBound, signFlag, signQuantDiff, dx_pos, dy_pos);

    free(currBlockRow);
    free(nextBlockRow);
    free(row_buffer);
    free(col_buffer);
    free(signQuantDiff);
    free(signFlag);
}

#endif