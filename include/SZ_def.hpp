#ifndef _SZ_DEF_HPP
#define _SZ_DEF_HPP

#include <cstdlib>
#include <cstring>

#define INT_BITS 32
#define INT_BYTES 4
#define FLOAT_BYTES 4
#define FIXED_RATE_PER_BLOCK_BYTES 1
#define REG_COEFF_SIZE_2D 3
#define REG_COEFF_SIZE_3D 4

std::string data_file_2d = "/Users/xuanwu/github/datasets/CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat";
std::string data_file_3d = "/Users/xuanwu/github/datasets/NYX/data/VelocityX.f32";
// std::string data_file_3d = "/Users/xuanwu/github/datasets/Hurricane/data/VelocityX.f32";

enum class appType
{
    MEAN,
    CENTRALDIFF,
    HEATDIS
};

enum class decmpState
{
	prePred,
    postPred
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
	size_t dim0_offset;
	DSize_2d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		max_num_block_elements = bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2;
		dim0_offset = r2;
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
	size_t dim0_offset;
	size_t dim1_offset;
	DSize_3d(size_t r1, size_t r2, size_t r3, int bs){
		dim1 = r1, dim2 = r2, dim3 = r3;
		nbEle = r1 * r2 * r3;
		Bsize = bs;
		max_num_block_elements = bs * bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		block_dim3 = (r3 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2 * block_dim3;
		dim0_offset = r2 * r3;
		dim1_offset = r3;
	}
};

template <class T>
inline int SZ_quantize(const T& data, double errorBound)
{
    return static_cast<int>(std::floor((data + errorBound) / (2 * errorBound)));
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
