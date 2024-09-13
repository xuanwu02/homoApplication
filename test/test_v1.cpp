#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "heatdis_entry.hpp"

void initData(int dim1, int dim2, float *h) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            h[i * dim2 + j] = 20;
        }
    }
}

void doWork(int dim1, int dim2, float *g, float *h)
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

int main(int argc, char **argv)
{
    int argv_id = 1;
    size_t dim1 = atoi(argv[argv_id++]);
    size_t dim2 = atoi(argv[argv_id++]);
    int blockSize = dim2;
    double errorBound = atof(argv[argv_id++]);
    int max_iter = atoi(argv[argv_id++]);
    size_t nbEle = dim1 * dim2;

    struct timespec start, end;
    double elapsed_time1, elapsed_time2;
    double max_err1, max_err2;

    float * oriData = (float *)calloc(nbEle, sizeof(float));
    initData(dim1, dim2, oriData);
    unsigned char *cmpData1 = (unsigned char *)calloc(2 * nbEle, sizeof(unsigned char));
    size_t cmpSize = 0;
    SZp_compress_1D(oriData, cmpData1, dim1, dim2, errorBound, blockSize, &cmpSize);
    unsigned char *cmpData2 = (unsigned char *)calloc(2 * nbEle, sizeof(unsigned char));
    memcpy(cmpData2, cmpData1, 2 * nbEle * sizeof(unsigned char));

    float * h = (float *)calloc(nbEle, sizeof(float));
    for(int i=0; i<max_iter; i++){
        doWork(dim1, dim2, oriData, h);
    }

    // DOC
    float * decData1 = (float *)malloc(nbEle * sizeof(float));
    clock_gettime(CLOCK_REALTIME, &start);
    for(int i=0; i<max_iter; i++){
        SZp_decompress_1D(decData1, cmpData1, dim1, dim2, errorBound, blockSize);
        doWork(dim1, dim2, decData1, h);
        SZp_compress_1D(decData1, cmpData1, dim1, dim2, errorBound, blockSize, &cmpSize);
    }
    clock_gettime(CLOCK_REALTIME, &end);
    SZp_decompress_1D(decData1, cmpData1, dim1, dim2, errorBound, blockSize);
    max_err1 = verify(oriData, decData1, dim1, dim2);
    elapsed_time1 = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("cr1 = %f, max_err1 = %.14f, elapsed_time1 = %.6f\n", 1.0 * sizeof(float) * nbEle / cmpSize, max_err1, elapsed_time1);

    // recover to quantization index
    clock_gettime(CLOCK_REALTIME, &start);
    SZp_heatdis_decompressToQuant(cmpData2, dim1, dim2, errorBound, blockSize, &cmpSize, max_iter);
    clock_gettime(CLOCK_REALTIME, &end);
    float * decData2 = (float *)malloc(nbEle * sizeof(float));
    SZp_decompress_1D(decData2, cmpData2, dim1, dim2, errorBound, blockSize);
    max_err2 = verify(oriData, decData2, dim1, dim2);
    elapsed_time2 = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
    printf("cr2 = %f, max_err2 = %.14f, elapsed_time2 = %.6f\n", 1.0 * sizeof(float) * nbEle / cmpSize, max_err2, elapsed_time2);
    // print
    print_matrix_int(dim1, dim2, "quant index2", qinds2);

    // print_matrix_float(dim1, dim2, "oriData", oriData);
    // print_matrix_float(dim1, dim2, "decData1", decData1);
    // print_matrix_float(dim1, dim2, "decData2", decData2);

    free(oriData);
    free(cmpData1);
    free(cmpData2);
    free(h);
    free(decData1);
    free(decData2);

    return 0;
}
