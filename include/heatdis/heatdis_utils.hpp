#ifndef _HEATDIS_UTILS_HPP
#define _HEATDIS_UTILS_HPP

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

const int INT_BITS = 32;

const double SRC_TEMP = 100.0;
const double WALL_TEMP = 0.0;
const double BACK_TEMP = 0.0;

void initData(int dim1, int dim2, float *h) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            h[i * dim2 + j] = 0;
        }
    }
}

void doWork(int dim1, int dim2, float * g, float * h)
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

void doWork(int dim1, int dim2, int max_iter, float * g, float * h)
{
    for(int i=0; i<max_iter; i++){
        doWork(dim1, dim2, g, h);
    }    
}


void compute_quant(int dim1, int dim2, int * quant, int * pred)
{
    int index;
    for(int i=0; i<dim1; i++){
        int prefix_sum = 0;
        for(int j=0; j<dim2; j++){
            index = i * dim2 + j;
            prefix_sum += pred[index];
            quant[index] = prefix_sum;
        }
    }
}

void compute_pred(int dim1, int dim2, int * quant, int * pred)
{
    int curr_quant, index;
    for(int i=0; i<dim1; i++){
        int prev_quant = 0;
        for(int j=0; j<dim2; j++){
            index = i * dim2 + j;
            curr_quant = quant[index];
            pred[index] = curr_quant - prev_quant;
            prev_quant = curr_quant;
        }
    }
}

double verify(const float *oriData, const float *decData, size_t dim1, size_t dim2)
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
    printf("max error position: (%lu, %lu)\n", pos/dim2 + 1, pos%dim2 + 1);
    return max_error;
}

void print_matrix_float(int dim1, int dim2, std::string name, float *mat)
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

inline int integerize_vanilla(int index, float *oriData, double errorBound, int& prev_quant)
{
    int curr_quant = static_cast<int>(std::floor((oriData[index] + errorBound) / (2 * errorBound)));
    int lorenzo_pred = curr_quant - prev_quant;
    prev_quant = curr_quant;
    return lorenzo_pred;
}

size_t compute_encoding_byteLength(size_t intArrayLength, int bit_count)
{
    unsigned int byte_count = bit_count / 8;
    unsigned int remainder_bit = bit_count % 8;
    size_t byteLength = byte_count * intArrayLength + (remainder_bit * intArrayLength - 1) / 8 + 1;
    if(!bit_count) byteLength = 0;
    return byteLength;
}

#endif
