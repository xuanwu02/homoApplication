#ifndef _COMP_UTILS_HPP
#define _COMP_UTILS_HPP

#include <cstdint>
#include <iostream>
#include <cstddef>
#include <filesystem>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <random>
#include "SZ_def.hpp"

template <class T>
void print_statistics(const T * data_ori, const T * data_dec, size_t data_size){
    double max_val = data_ori[0];
    double min_val = data_ori[0];
    double max_abs = fabs(data_ori[0]);
    for(int i=0; i<data_size; i++){
        if(data_ori[i] > max_val) max_val = data_ori[i];
        if(data_ori[i] < min_val) min_val = data_ori[i];
        if(fabs(data_ori[i]) > max_abs) max_abs = fabs(data_ori[i]);
    }
    double max_err = 0;
    int pos = 0;
    double mse = 0;
    for(int i=0; i<data_size; i++){
        double err = data_ori[i] - data_dec[i];
        mse += err * err;
        if(fabs(err) > max_err){
            pos = i;
            max_err = fabs(err);
        }
    }
    mse /= data_size;
    double psnr = 20 * log10((max_val - min_val) / sqrt(mse));
    std::cout << "Max value = " << max_val << ", min value = " << min_val << std::endl;
    std::cout << "Max error = " << max_err << ", pos = " << pos << std::endl;
    std::cout << "MSE = " << mse << ", PSNR = " << psnr << std::endl;
}

template<typename Type>
void writefile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
    fout.close();
}

template<typename Type>
std::vector<Type> readfile(const char *file, size_t &num) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cout << " Error, Couldn't find the file" << "\n";
        return std::vector<Type>();
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    auto data = std::vector<Type>(num_elements);
    fin.read(reinterpret_cast<char *>(&data[0]), num_elements * sizeof(Type));
    fin.close();
    num = num_elements;
    return data;
}

template<class T>
void read(T &var, unsigned char const *&compressed_data_pos) {
    memcpy(&var, compressed_data_pos, sizeof(T));
    compressed_data_pos += sizeof(T);
}

template<class T>
void write(T const var, unsigned char *&compressed_data_pos) {
    memcpy(compressed_data_pos, &var, sizeof(T));
    compressed_data_pos += sizeof(T);
}

std::vector<std::string> getFiles(const std::string& dirPath)
{
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dirPath))
    {
        if (entry.is_regular_file())
        {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

template<class T>
void set_relative_eb(const std::vector<T>& oriData_vec, double& errorBound){
    auto max_val = *std::max_element(oriData_vec.begin(), oriData_vec.end());
    auto min_val = *std::min_element(oriData_vec.begin(), oriData_vec.end());
    auto range = max_val - min_val;
    errorBound *= range;
    printf("Max = %.4e, min = %.4e, range = %.6e, abs_eb = %.6e\n", max_val, min_val, range, errorBound);
}

template <class T>
void set_overall_eb(const T* vx, const T* vy, size_t n, double& eb){
    std::vector<T> v2(2*n, 0);
    memcpy(v2.data(), vx, n*sizeof(T));
    memcpy(v2.data()+n, vy, n*sizeof(T));
    set_relative_eb(v2, eb);
}

template <class T>
void set_overall_eb(const T* vx, const T* vy, const T* vz, size_t n, double& eb){
    std::vector<T> v3(3*n, 0);
    memcpy(v3.data(), vx, n*sizeof(T));
    memcpy(v3.data()+n, vy, n*sizeof(T));
    memcpy(v3.data()+n*2, vz, n*sizeof(T));
    set_relative_eb(v3, eb);
}

double get_elapsed_time(struct timespec &start, struct timespec &end){
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
}

template <class T>
void initRandomData(T min, T max, unsigned int seed, size_t n, T *data){
    std::mt19937 generator(seed);  
    std::uniform_real_distribution<T> distribution(min, max);
    for(size_t i=0; i<n; i++){
        data[i] = distribution(generator);
    }
}

template <class T>
double verify(const T *oriData, const T *decData, size_t dim1, size_t dim2)
{
    size_t n = dim1 * dim2;
    int pos = 0;
    double max_error = 0;
    for(size_t i=0; i<n; i++){
        double diff = fabs(oriData[i] - decData[i]);
        if(diff > max_error){
            max_error = diff;
            pos = i;
        }  
    }
    // std::cout << "max_error = " << max_error << ", pos = " << pos << ", x = " << pos / dim2 << ", y = " << pos % dim2 << std::endl;
    return max_error;
}

template <class T>
double verify(const T *oriData, const T *decData, size_t dim1, size_t dim2, size_t dim3)
{
    size_t n = dim1 * dim2 * dim3;
    int pos = 0;
    double max_error = 0;
    for(size_t i=0; i<n; i++){
        double diff = fabs(oriData[i] - decData[i]);
        if(diff > max_error){
            max_error = diff;
            pos = i;
        }  
    }
    // std::cout << "max_error = " << max_error << ", pos = " << pos << ", x = " << pos / (dim2 * dim3) << ", y = " << (pos % (dim2 * dim3)) / dim3 << ", z = " << pos % dim3 << std::endl;
    return max_error;
}

template <class T>
double verify_dxdy(const T *oriData, const T *decData, size_t dim1, size_t dim2)
{
    size_t x, y;
    double max_error = 0;
    double v1, v2;
    for(size_t i=1; i<dim1-1; i++){
        const T * ori_pos = oriData + i * dim2;
        const T * dec_pos = decData + i * dim2;
        for(size_t j=1; j<dim2-1; j++){
            double diff = fabs(ori_pos[j] - dec_pos[j]);
            if(diff > max_error){
                max_error = diff;
                x = i, y = j;
                v1 = ori_pos[j];
                v2 = dec_pos[j];
            }
        }
    }
    // std::cout << "max_error = " << max_error << ", x = " << x << ", y = " << y << std::endl;
    // std::cout << "ori = " << v1 << ", dec = " << v2 << std::endl;
    return max_error;
}

template <class T>
double verify_dxdydz(const T *oriData, const T *decData, size_t dim1, size_t dim2, size_t dim3)
{
    size_t dim0_offset = dim2 * dim3;
    size_t dim1_offset = dim3;
    size_t x, y, z;
    double max_error = 0;
    double v1, v2;
    for(size_t i=1; i<dim1-1; i++){
        const T * x_ori_pos = oriData + i * dim0_offset;
        const T * x_dec_pos = decData + i * dim0_offset;
        for(size_t j=1; j<dim2-1; j++){
            const T * y_ori_pos = x_ori_pos + j * dim1_offset;
            const T * y_dec_pos = x_dec_pos + j * dim1_offset;
            for(size_t k=1; k<dim3-1; k++){
                double diff = fabs(y_ori_pos[k] - y_dec_pos[k]);
                if(diff > max_error){
                    max_error = diff;
                    x = i, y = j, z = k;
                    v1 = y_ori_pos[k];
                    v2 = y_dec_pos[k];
                }
            }
        }
    }
    // std::cout << "max_error = " << max_error << ", x = " << x << ", y = " << y << ", z = " << z << std::endl;
    // std::cout << "ori = " << v1 << ", dec = " << v2 << std::endl;
    return max_error;
}

template <class T>
void print_matrix_float(int dim1, int dim2, std::string name, T *mat)
{
    std::cout << "--------- " << name << " ---------" << std::endl;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%.8f  ", mat[i*dim2+j]);
        }
        printf("\n");
    }
}

void print_matrix_int(int dim1, int dim2, std::string name, int *mat)
{
    std::cout << "--------- " << name << " ---------" << std::endl;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%d ", mat[i*dim2+j]);
        }
        printf("\n");
    }
}

template <class T>
void compute_dxdy(
    size_t dim1, size_t dim2, T *data,
    T *dx_result, T *dy_result
){
    T *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    size_t i, j;
    for(i=1; i<dim1-1; i++){
        curr_row = data + i * dim2;
        prev_row = curr_row - dim2;
        next_row = curr_row + dim2;
        dx_pos = dx_result + i * dim2;
        dy_pos = dy_result + i * dim2;
        for(j=1; j<dim2-1; j++){
            dx_pos[j] = (next_row[j] - prev_row[j]) * 0.5;
            dy_pos[j] = (curr_row[j+1] - curr_row[j-1]) * 0.5;
        }
    }
}

template <class T>
void compute_dxdydz(
    size_t dim1, size_t dim2, size_t dim3, T *data,
    T *dx_result, T *dy_result, T *dz_result
){
    size_t dim0_offset = dim2 * dim3;
    size_t dim1_offset = dim3;
    size_t i, j, k;
    T *curr_plane = nullptr, *prev_plane = nullptr, *next_plane = nullptr;
    T *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *x_dx_pos = nullptr, *x_dy_pos = nullptr, *x_dz_pos = nullptr;
    T *y_dx_pos = nullptr, *y_dy_pos = nullptr, *y_dz_pos = nullptr;
    for(i=1; i<dim1-1; i++){
        curr_plane = data + i * dim0_offset;
        prev_plane = curr_plane - dim0_offset;
        next_plane = curr_plane + dim0_offset;
        curr_row = curr_plane + dim1_offset;
        x_dx_pos = dx_result + i * dim0_offset;
        x_dy_pos = dy_result + i * dim0_offset;
        x_dz_pos = dz_result + i * dim0_offset;
        for(j=1; j<dim2-1; j++){
            prev_row = curr_row - dim1_offset;
            next_row = curr_row + dim1_offset;
            y_dx_pos = x_dx_pos + j * dim1_offset;
            y_dy_pos = x_dy_pos + j * dim1_offset;
            y_dz_pos = x_dz_pos + j * dim1_offset;
            for(k=1; k<dim3-1; k++){
                size_t index_1d = k;
                size_t index_2d = j * dim1_offset + k;
                y_dx_pos[index_1d] = (next_plane[index_2d] - prev_plane[index_2d]) * 0.5;
                y_dy_pos[index_1d] = (next_row[index_1d] - prev_row[index_1d]) * 0.5;
                y_dz_pos[index_1d] = (curr_row[index_1d + 1] - curr_row[index_1d - 1]) * 0.5;
            }
            curr_row += dim1_offset;
        }
    }
}

template <class T>
void compute_divergence_2d(
    size_t nx, size_t ny,
    const T* vx, const T* vy, T* result
){
    auto idx = [&](size_t i, size_t j) {
        return i * ny + j;
    };
    for (size_t i = 1; i < nx - 1; i++) {
        for (size_t j = 1; j < ny - 1; j++) {
            T dfx = vx[idx(i + 1, j)] - vx[idx(i - 1, j)];
            T dfy = vy[idx(i, j + 1)] - vy[idx(i, j - 1)];
            result[idx(i, j)] = (dfx + dfy) * 0.5;
        }
    }
}

template <typename T>
void compute_divergence_3d(
    size_t nx, size_t ny, size_t nz,
    const T* vx, const T* vy, const T* vz, T* result
){
    auto idx = [&](size_t i, size_t j, size_t k) {
        return i * ny * nz + j * nz + k;
    };
    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t k = 1; k < nz - 1; ++k) {
                T dfx = vx[idx(i + 1, j, k)] - vx[idx(i - 1, j, k)];
                T dfy = vy[idx(i, j + 1, k)] - vy[idx(i, j - 1, k)];
                T dfz = vz[idx(i, j, k + 1)] - vz[idx(i, j, k - 1)];
                result[idx(i, j, k)] = (dfx + dfy + dfz) * 0.5;
            }
        }
    }
}

template <typename T>
void compute_curl_2d(
    size_t nx, size_t ny, const T* vx, const T* vy, T* curl
){
    auto idx = [&](size_t i, size_t j) {
        return i * ny + j;
    };
    for (size_t i = 1; i < nx - 1; i++) {
        for (size_t j = 1; j < ny - 1; j++) {
            curl[idx(i, j)] = ((vy[idx(i + 1, j)] - vy[idx(i - 1, j)]) - (vx[idx(i, j + 1)] - vx[idx(i, j - 1)])) * 0.5;
        }
    }
}

template <typename T>
void compute_curl_3d(
    size_t nx, size_t ny, size_t nz,
    const T* vx, const T* vy, const T* vz,
    T* curlx, T* curly, T* curlz
){
    auto idx = [&](size_t i, size_t j, size_t k) {
        return i * ny * nz + j * nz + k;
    };
    for (size_t i = 1; i < nx - 1; ++i) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t k = 1; k < nz - 1; ++k) {
                curlx[idx(i, j, k)] = ((vz[idx(i, j + 1, k)] - vz[idx(i, j - 1, k)]) - (vy[idx(i, j, k + 1)] - vy[idx(i, j, k - 1)])) * 0.5;
                curly[idx(i, j, k)] = ((vx[idx(i, j, k + 1)] - vx[idx(i, j, k - 1)]) - (vz[idx(i + 1, j, k)] - vz[idx(i - 1, j, k)])) * 0.5;
                curlz[idx(i, j, k)] = ((vy[idx(i + 1, j, k)] - vy[idx(i - 1, j, k)]) - (vx[idx(i, j + 1, k)] - vx[idx(i, j - 1, k)])) * 0.5;
            }
        }
    }
}

template <class T>
void compute_laplacian_2d(
    size_t dim1, size_t dim2, T *data, T *result
){
    T *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *result_pos = nullptr;
    size_t i, j;
    for(i=1; i<dim1-1; i++){
        curr_row = data + i * dim2;
        prev_row = curr_row - dim2;
        next_row = curr_row + dim2;
        result_pos = result + i * dim2;
        for(j=1; j<dim2-1; j++){
            result_pos[j] = curr_row[j-1] + curr_row[j+1] + prev_row[j] + next_row[j] - 4 * curr_row[j];
        }
    }
}

template <class T>
void compute_laplacian_3d(
    size_t dim1, size_t dim2, size_t dim3, T *data, T *result
){
    size_t dim0_offset = dim2 * dim3;
    size_t dim1_offset = dim3;
    size_t i, j, k;
    T *curr_plane = nullptr, *prev_plane = nullptr, *next_plane = nullptr;
    T *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *x_res_pos = nullptr;
    T *y_res_pos = nullptr;
    for(i=1; i<dim1-1; i++){
        curr_plane = data + i * dim0_offset;
        prev_plane = curr_plane - dim0_offset;
        next_plane = curr_plane + dim0_offset;
        curr_row = curr_plane + dim1_offset;
        x_res_pos = result + i * dim0_offset;
        for(j=1; j<dim2-1; j++){
            prev_row = curr_row - dim1_offset;
            next_row = curr_row + dim1_offset;
            y_res_pos = x_res_pos + j * dim1_offset;
            for(k=1; k<dim3-1; k++){
                size_t index_1d = k;
                size_t index_2d = j * dim1_offset + k;
                y_res_pos[index_1d] = curr_row[k-1] + curr_row[k+1] +
                                      prev_row[k] + next_row[k] +
                                      prev_plane[index_2d] + next_plane[index_2d] -
                                      6 * curr_row[k];
            }
            curr_row += dim1_offset;
        }
    }
}

template <class T>
double compute_region_mean(
    size_t dim1, size_t dim2, int blockSideLength, double ratio, T *data
){
    DSize_2d size(dim1, dim2, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * size.Bsize * size.Bsize;
    dlo1 = lo1 * size.Bsize;
    dhi1 = hi1 * size.Bsize;
    dlo2 = lo2 * size.Bsize;
    dhi2 = hi2 * size.Bsize;
    double sum = 0;
    T * x_data_pos = data + dlo1 * size.offset_0;
    for(size_t i=dlo1; i<dhi1; i++){
        T * y_data_pos = x_data_pos;
        for(size_t j=dlo2; j<dhi2; j++){
            sum += y_data_pos[j];
        }
        x_data_pos += size.offset_0;
    }
    return sum / (double)region_size;
}

template <class T>
double compute_region_mean(
    size_t dim1, size_t dim2, size_t dim3, int blockSideLength, double ratio, T *data
){
    DSize_3d size(dim1, dim2, dim3, blockSideLength);
    size_t dlo1 = floor(dim1 * (1.0 - ratio) * 0.5);
    size_t dhi1 = floor(dim1 * (1.0 + ratio) * 0.5);
    size_t dlo2 = floor(dim2 * (1.0 - ratio) * 0.5);
    size_t dhi2 = floor(dim2 * (1.0 + ratio) * 0.5);
    size_t dlo3 = floor(dim3 * (1.0 - ratio) * 0.5);
    size_t dhi3 = floor(dim3 * (1.0 + ratio) * 0.5);
    size_t lo1 = dlo1 / size.Bsize;
    size_t hi1 = dhi1 / size.Bsize + 1;
    size_t lo2 = dlo2 / size.Bsize;
    size_t hi2 = dhi2 / size.Bsize + 1;
    size_t lo3 = dlo3 / size.Bsize;
    size_t hi3 = dhi3 / size.Bsize + 1;
    size_t region_size = (hi1 - lo1) * (hi2 - lo2) * (hi3 - lo3) * size.Bsize * size.Bsize * size.Bsize;
    dlo1 = lo1 * size.Bsize;
    dhi1 = hi1 * size.Bsize;
    dlo2 = lo2 * size.Bsize;
    dhi2 = hi2 * size.Bsize;
    dlo3 = lo3 * size.Bsize;
    dhi3 = hi3 * size.Bsize;
    double sum = 0;
    T * x_data_pos = data + dlo1 * size.offset_0;
    for(size_t i=dlo1; i<dhi1; i++){
        T * y_data_pos = x_data_pos + dlo2 * size.offset_1;
        for(size_t j=dlo2; j<dhi2; j++){
            for(size_t k=dlo3; k<dhi3; k++){
                sum += y_data_pos[k];
            }
            y_data_pos += size.offset_1;
        }
        x_data_pos += size.offset_0;
    }
    return sum / (double)region_size;
}

#endif