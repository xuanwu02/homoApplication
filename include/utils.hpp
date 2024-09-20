#ifndef _STENCIL_UTILS_HPP
#define _STENCIL_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

// modified from TypeManager.c
// change return value and increment byteArray
void 
convertIntArray2ByteArray_fast_1b_to_result_sz(const unsigned char* intArray, size_t intArrayLength, unsigned char *& compressed_pos){
    size_t byteLength = 0;
    size_t i, j; 
    if(intArrayLength%8==0)
        byteLength = intArrayLength/8;
    else
        byteLength = intArrayLength/8+1;
        
    size_t n = 0;
    int tmp, type;
    for(i = 0;i<byteLength;i++){
        tmp = 0;
        for(j = 0;j<8&&n<intArrayLength;j++){
            type = intArray[n];
            if(type == 1)
                tmp = (tmp | (1 << (7-j)));
            n++;
        }
        *(compressed_pos++) = (unsigned char)tmp;
    }
}

// modified from TypeManager.c
// change return value and increment compressed_pos
unsigned char * 
convertByteArray2IntArray_fast_1b_sz(size_t intArrayLength, const unsigned char*& compressed_pos, size_t byteArrayLength){
    if(intArrayLength > byteArrayLength*8){
        printf("Error: intArrayLength > byteArrayLength*8\n");
        printf("intArrayLength=%zu, byteArrayLength = %zu", intArrayLength, byteArrayLength);
        exit(0);
    }
    unsigned char * intArray = NULL;
    if(intArrayLength>0) intArray = (unsigned char*)malloc(intArrayLength*sizeof(unsigned char));
    size_t n = 0, i;
    int tmp;
    for (i = 0; i < byteArrayLength-1; i++) {
        tmp = *(compressed_pos++);
        intArray[n++] = (tmp & 0x80) >> 7;
        intArray[n++] = (tmp & 0x40) >> 6;
        intArray[n++] = (tmp & 0x20) >> 5;
        intArray[n++] = (tmp & 0x10) >> 4;
        intArray[n++] = (tmp & 0x08) >> 3;
        intArray[n++] = (tmp & 0x04) >> 2;
        intArray[n++] = (tmp & 0x02) >> 1;
        intArray[n++] = (tmp & 0x01) >> 0;      
    }
    tmp = *(compressed_pos++);  
    for(int i=0; n < intArrayLength; n++, i++){
        intArray[n] = (tmp & (1 << (7 - i))) >> (7 - i);
    }       
    return intArray;
}

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

// katrina only
template <class T>
void print_statistics(const T * data_ori, const T * data_dec, size_t data_size, const T invalid_val){
    double max_val = INT8_MIN * 1.0;
    double min_val = INT8_MAX * 1.0;
    for(int i=0; i<data_size; i++){
        if(data_ori[i] > max_val && data_ori[i] != invalid_val) max_val = data_ori[i];
        if(data_ori[i] < min_val && data_ori[i] != invalid_val) min_val = data_ori[i];
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

template<typename Type>
void writefile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
    fout.close();
}

// normalize data to [0, 1]
inline double normalize_data(double data, double min, double range){
    return (data - min)/range;
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

// heatdis utils
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
    // printf("max error position: (%lu, %lu)\n", pos/dim2 + 1, pos%dim2 + 1);
    return max_error;
}

void print_matrix_float(int dim1, int dim2, std::string name, float *mat)
{
    std::cout << "--------- " << name << " ---------" << std::endl;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%.4f ", mat[i*dim2+j]);
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

#endif