#ifndef _SZ_DEF_HPP
#define _SZ_DEF_HPP

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>

#define INT_BITS 32
#define INT_BYTES 4
#define FLOAT_BYTES 4
#define FIXED_RATE_PER_BLOCK_BYTES 1

enum class appType
{
    STAT,
    DIFF
};

enum class decmpState
{
	full,
	meta,
	prePred,
    postPred
};

decmpState intToDecmpState(int value){
    switch(value){
        case 0: return decmpState::full;
        case 1: return decmpState::prePred;
        case 2: return decmpState::postPred;
        case 3: return decmpState::meta;
        default: 
            throw std::invalid_argument("Invalid integer for decmpState");
    }
}

struct DSize2D_1d
{
	size_t dim1;
	size_t dim2;
	size_t nbEle;
	int Bsize;
	int Bwidth;
	size_t num_blocks;
	size_t num_blockSlice;
	size_t offset_0;
	DSize2D_1d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		Bwidth = (int)std::sqrt(bs);
		num_blocks = (nbEle - 1) / bs + 1;
		num_blockSlice = (dim1 - 1) / Bwidth + 1;
		offset_0 = r2;
	}
};

struct DSize3D_1d
{
	size_t dim1;
	size_t dim2;
	size_t dim3;
	size_t nbEle;
	int Bsize;
	int Bwidth;
	size_t num_blocks;
	size_t num_blockSlice;
	size_t offset_0;
	size_t offset_1;
	DSize3D_1d(size_t r1, size_t r2, size_t r3, int bs){
		dim1 = r1, dim2 = r2, dim3 = r3;
		nbEle = r1 * r2 * r3;
		Bsize = bs;
		Bwidth = (int)std::sqrt(bs);
		num_blocks = (nbEle - 1) / bs + 1;
		num_blockSlice = (dim1 - 1) / Bwidth + 1;
		offset_0 = r2 * r3;
		offset_1 = r3;
	}
};

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
	size_t offset_0;
	DSize_2d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		max_num_block_elements = bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2;
		offset_0 = r2;
	}
};

struct DSize_3d
{
	size_t dim1;
	size_t dim2;
	size_t dim3;
	size_t nbEle;
	int Bsize;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t block_dim3;
	size_t num_blocks;
	size_t offset_0;
	size_t offset_1;
	DSize_3d(size_t r1, size_t r2, size_t r3, int bs){
		dim1 = r1, dim2 = r2, dim3 = r3;
		nbEle = r1 * r2 * r3;
		Bsize = bs;
		max_num_block_elements = bs * bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		block_dim3 = (r3 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2 * block_dim3;
		offset_0 = r2 * r3;
		offset_1 = r3;
	}
};

// template <class T>
// inline int SZ_quantize(const T& data, const double& errorBound)
// {
//     return static_cast<int>(std::floor((data + errorBound) / (2 * errorBound)));
// }
template <class T>
inline int SZ_quantize(const T& data, const double& inver_eb)
{
    return static_cast<int>(std::floor(data * inver_eb + 0.5));
}

template <class T>
inline void exchange_buffer(
    T *& curr, T *& next, T *& temp
){
    temp = curr;
    curr = next;
    next = temp;
}

template <class T>
inline void rotate_buffer(
    T *& curr, T *& prev, T *& next, T *& temp
){
    temp = prev;
    prev = curr;
    curr = next;
    next = temp;
}

#endif
