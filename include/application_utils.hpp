#ifndef _HOMO_APPLICATION_UTILS_HPP
#define _HOMO_APPLICATION_UTILS_HPP

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

const int INT_BITS = 32;

const double SRC_TEMP = 100.0;
const double WALL_TEMP = 0.0;
const double BACK_TEMP = 0.0;

const std::string cldhigh_data_file = "/Users/xuanwu/github/datasets/CESM-ATM-cleared-1800x3600/CLDHGH_1_1800_3600.dat";

/**
 * heatdis: initialize data (all zero)
*/
template <class T>
void initData(int dim1, int dim2, T *h)
{
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            h[i * dim2 + j] = 0;
        }
    }
}

template <class T>
void doWork(int dim1, int dim2, T * g, T * h)
{
    memcpy(h, g, dim1 * dim2 * sizeof(float));
    float left, right, up, down;
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            int index = i * dim2 + j;
            left = (j == 0) ? WALL_TEMP : h[index - 1];
            right = (j == dim2 - 1) ? WALL_TEMP : h[index + 1];
            up = (i == 0) ? SRC_TEMP : h[index - dim2];
            down = (i == dim1 - 1) ? BACK_TEMP : h[index + dim2];
            g[index] = 0.25 * (left + right + up + down);
        }
    }
}

template <class T>
void doWork(int dim1, int dim2, int max_iter, T * g, T * h)
{
    for(int i=0; i<max_iter; i++){
        doWork<T>(dim1, dim2, g, h);
    }    
}

// /**
//  * random data array generator
// */
// template <class T>
// void initRandomData(T min, T max, unsigned int seed, size_t n, T *data){
//     std::mt19937 generator(seed);  
//     std::uniform_real_distribution<T> distribution(min, max);
//     for(size_t i=0; i<n; i++){
//         data[i] = distribution(generator);
//     }
// }

/**
 * central difference for 2D data
*/
template <class T>
void compute_dxdy(
    int dim1, int dim2, T *data,
    T *dx_result, T *dy_result
){
    T *curr_row = nullptr, *prev_row = nullptr, *next_row = nullptr;
    T *dx_pos = nullptr, *dy_pos = nullptr;
    int i, j;
    {
        i = 0;
        curr_row = data + i * dim2;
        next_row = data + (i + 1) * dim2;
        dx_pos = dx_result + i * dim2;
        dy_pos = dy_result + i * dim2;
        {
            j = 0;
            dx_pos[j] = curr_row[j + 1] - curr_row[j];
            dy_pos[j] = next_row[j] - curr_row[j];
        }
        for(j=1; j<dim2-1; j++){
            dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * 0.5;
            dy_pos[j] = next_row[j] - curr_row[j];
        }
        {
            j = dim2 - 1;
            dx_pos[j] = (curr_row[j] - curr_row[j - 1]);
            dy_pos[j] = (next_row[j] - curr_row[j]);
        }
    }
    for(i=1; i<dim1-1; i++){
        curr_row = data + i * dim2;
        prev_row = data + (i - 1) * dim2;
        next_row = data + (i + 1) * dim2;
        dx_pos = dx_result + i * dim2;
        dy_pos = dy_result + i * dim2;
        {
            j = 0;
            dx_pos[j] = curr_row[j + 1] - curr_row[j];
            dy_pos[j] = (next_row[j] - prev_row[j]) * 0.5;
        }
        for(j=1; j<dim2-1; j++){
            dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * 0.5;
            dy_pos[j] = (next_row[j] - prev_row[j]) * 0.5;
        }
        {
            j = dim2 - 1;
            dx_pos[j] = curr_row[j] - curr_row[j - 1];
            dy_pos[j] = (next_row[j] - prev_row[j]) * 0.5;
        }
    }
    {
        i = dim1 - 1;
        curr_row = data + i * dim2;
        prev_row = data + (i - 1) * dim2;
        dx_pos = dx_result + i * dim2;
        dy_pos = dy_result + i * dim2;
        {
            j = 0;
            dx_pos[j] = curr_row[j + 1] - curr_row[j];
            dy_pos[j] = curr_row[j] - prev_row[j];
        }
        for(j=1; j<dim2-1; j++){
            dx_pos[j] = (curr_row[j + 1] - curr_row[j - 1]) * 0.5;
            dy_pos[j] = curr_row[j] - prev_row[j];
        }
        {
            j = dim2 - 1;
            dx_pos[j] = curr_row[j] - curr_row[j - 1];
            dy_pos[j] = curr_row[j] - prev_row[j];
        }
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
    return max_error;
}

/**
 * for float & double arrays
*/
template <class T>
void print_matrix_float(int dim1, int dim2, std::string name, T *mat)
{
    std::cout << "--------- " << name << " ---------" << std::endl;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%.2f  ", mat[i*dim2+j]);
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
inline int SZp_quantize(const T& data, double errorBound)
{
    return static_cast<int>(std::floor((data + errorBound) / (2 * errorBound)));
}

#endif
