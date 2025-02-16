#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cassert>
#include "SZp_LorenzoPredictor3D.hpp"
#include "utils.hpp"
#include "settings.hpp"

int main(int argc, char **argv)
{
    int argv_id = 1;
    std::string config(argv[argv_id++]);
    std::string data_file(argv[argv_id++]);
    Settings s = Settings::from_json(config);
    int opt = atoi(argv[argv_id++]);

    using T = float;

    size_t nbEle;
    auto oriData_vec = readfile<T>(data_file.c_str(), nbEle);
    assert(nbEle == s.dim1 * s.dim2 * s.dim3);
    T * oriData = oriData_vec.data();
    set_relative_eb(oriData_vec, s.eb);

    // int * decQuant = (int *)malloc(nbEle * sizeof(int));
    int * decQuant = (int *)malloc((s.dim1+1) * (s.dim2+1) * (s.dim3+1) * sizeof(int));
    unsigned char *cmpData = (unsigned char *)malloc(nbEle * sizeof(T));
    T * decData = (T *)malloc(nbEle * sizeof(T));

    struct timespec start, end;
    double elapsed_time;

    size_t cmpSize = 0;
    SZp_compress_3dLorenzo(oriData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb, cmpSize);
    printf("cr = %.2f\n", 1.0 * nbEle * sizeof(T) / cmpSize);

    switch(opt){
        case 0:{
            clock_gettime(CLOCK_REALTIME, &start);
            SZp_decompress_3dLorenzo(decData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb);
            clock_gettime(CLOCK_REALTIME, &end);
            elapsed_time = get_elapsed_time(start, end);
            printf("full decompression elapsed_time = %.6f\n", elapsed_time);
            break;
        }
        case 1:{
            clock_gettime(CLOCK_REALTIME, &start);
            SZp_decompress_to_PrePred_3dLorenzo(decQuant, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb);
            clock_gettime(CLOCK_REALTIME, &end);
            elapsed_time = get_elapsed_time(start, end);
            printf("decompress to PrePred elapsed_time = %.6f\n", elapsed_time);
            SZp_decompress_3dLorenzo(decData, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb);
            T max_err = 0;
            // for(int i=0; i<s.dim1*s.dim2*s.dim3; i++){
            //     T err = decQuant[i] * 2 * s.eb - decData[i];
            //     if(fabs(err) > max_err) max_err = fabs(err);
            // }
            int index = 0;
            for(int i=1; i<s.dim1+1; i++){
                for(int j=1; j<s.dim2+1; j++){
                    for(int k=1; k<s.dim3+1; k++){
                        T err = decQuant[i*(s.dim2+1)*(s.dim3+1)+j*(s.dim3+1)+k] * 2 * s.eb - decData[index];
                        if(fabs(err) > max_err) max_err = fabs(err);
                        index ++;
                    }
                }
            }
            printf("eb = %.6f, max error = %.6f\n", s.eb, max_err);
            break;
        }
        case 2:{
            clock_gettime(CLOCK_REALTIME, &start);
            SZp_decompress_to_PostPred_3dLorenzo(decQuant, cmpData, s.dim1, s.dim2, s.dim3, s.B, s.eb);
            clock_gettime(CLOCK_REALTIME, &end);
            elapsed_time = get_elapsed_time(start, end);
            printf("decompress to PostPred elapsed_time = %.6f\n", elapsed_time);
            break;
        }
        default:
            break;
    }

    free(decQuant);
    free(decData);
    free(cmpData);

    return 0;
}
