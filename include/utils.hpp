#ifndef _HOMOAPP_UTILS_HPP
#define _HOMOAPP_UTILS_HPP

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

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

#endif