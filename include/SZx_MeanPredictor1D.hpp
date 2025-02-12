#ifndef _SZX_MEAN_PREDICTOR_1D_HPP
#define _SZX_MEAN_PREDICTOR_1D_HPP

#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "typemanager.hpp"
#include "SZx_app_utils.hpp"
#include "utils.hpp"
#include "settings.hpp"

template <class T>
void SZx_compress_1dMeanbased(
    const T *oriData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound, size_t& cmpSize
){
    const DSize_1d size(dim1, dim2, blockSideLength);
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements*sizeof(unsigned int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * block_quant_inds = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    const T * x_data_pos = oriData;
    int block_ind = 0;
    for(size_t x=0; x<size.block_dim1; x++){
        const T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int fixed_rate, max_err = 0;
            unsigned int * abs_err_pos = absPredError;
            unsigned char * sign_pos = signFlag;
            int * block_buffer_pos = block_quant_inds;
            int mean_quant = compute_block_mean_quant(block_size, y_data_pos, block_buffer_pos, errorBound);
            for(int i=0; i<block_size; i++){
                int err = *block_buffer_pos++ - mean_quant;
                int abs_err = abs(err);
                *sign_pos++ = (err < 0);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            for(int k=3; k>=0; k--){
                *(qmean_pos++) = (mean_quant >> (8 * k)) & 0xff;
            }
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
    cmpSize = encode_pos - cmpData;
    free(absPredError);
    free(signFlag);
    free(block_quant_inds);
}

template <class T>
void SZx_decompress_1dMeanbased(
    T *decData, unsigned char *cmpData,
    size_t dim1, size_t dim2, int blockSideLength,
    double errorBound
){
    const DSize_1d size(dim1, dim2, blockSideLength);
    int * signPredError = (int *)malloc(size.max_num_block_elements*sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements*sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    T * x_data_pos = decData;
    int block_ind = 0;
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, blocks_mean_quant, size.num_blocks);
    for(size_t x=0; x<size.block_dim1; x++){
        T * y_data_pos = x_data_pos;
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int mean_quant = blocks_mean_quant[block_ind];
            int fixed_rate = (int)cmpData[block_ind++];
            T * curr_data_pos = y_data_pos;
            if(fixed_rate){
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(signFlag, signPredError, block_size);
                int * pred_err_pos = signPredError;
                for(int i=0; i<block_size; i++){
                    *curr_data_pos++ = (*pred_err_pos++ + mean_quant) * 2 * errorBound;
                }
            }else{
                for(int i=0; i<block_size; i++){
                    *curr_data_pos++ = mean_quant * 2 * errorBound;
                }
            }
            y_data_pos += size.Bsize;
        }
        x_data_pos += size.dim0_offset;
    }
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
}

// heatdis
inline void recoverBlockRow2PrePred(
    size_t x, DSize_1d size, SZxCmpBufferSet *cmpkit_set,
    unsigned char *& encode_pos, int *buffer_data_pos, int current,
    size_t buffer_dim0_offset
){
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int block_ind = x * size.Bwidth * size.block_dim2;
    int * buffer_start_pos = buffer_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            int * curr_buffer_pos = buffer_start_pos;
            int fixed_rate = (int)cmpkit_set->cmpData[current][block_ind];
            int mean_quant = cmpkit_set->mean_quant_inds[block_ind];
            block_ind++;
            if(!fixed_rate){
                for(int j=0; j<block_size; j++){
                    curr_buffer_pos[j] = mean_quant;
                }
            }
            else{
                size_t cmp_block_sign_length = (block_size + 7) / 8;
                convertByteArray2IntArray_fast_1b_args(block_size, encode_pos, cmp_block_sign_length, cmpkit_set->signFlag);
                encode_pos += cmp_block_sign_length;
                unsigned int savedbitsbytelength = Jiajun_extract_fixed_length_bits(encode_pos, block_size, cmpkit_set->signPredError, fixed_rate);
                encode_pos += savedbitsbytelength;
                convert2SignIntArray(cmpkit_set->signFlag, cmpkit_set->signPredError, block_size);
                int * data_pos = cmpkit_set->signPredError;
                for(int j=0; j<block_size; j++){
                    curr_buffer_pos[j] = data_pos[j] + mean_quant;
                }
            }        
            buffer_start_pos += size.Bsize;
        }
        buffer_start_pos += buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
}

inline void heatdisProcessCompressBlockRowPrePred(
    size_t x, DSize_1d size, TempInfo2D& temp_info,
    SZxCmpBufferSet *cmpkit_set, SZxAppBufferSet_2d *buffer_set,
    double errorBound, int iter, int next,
    bool isTopRow, bool isBottomRow
){
    int bias = (iter & 1) + 1;
    int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
    int block_ind = x * size.Bwidth * size.block_dim2;
    const int * prevBlockRowBottom_pos = isTopRow ? nullptr : buffer_set->prevRow_data_pos + (size.Bwidth - 1) * buffer_set->buffer_dim0_offset - 1;
    const int * nextBlockRowTop_pos = isBottomRow ? nullptr : buffer_set->nextRow_data_pos - 1;
    if(!isTopRow) memcpy(buffer_set->currRow_data_pos-buffer_set->buffer_dim0_offset-1, prevBlockRowBottom_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    if(!isBottomRow) memcpy(buffer_set->currRow_data_pos+size.Bwidth*buffer_set->buffer_dim0_offset-1, nextBlockRowTop_pos, buffer_set->buffer_dim0_offset*sizeof(int));
    set_buffer_border_prepred(buffer_set->currRow_data_pos, size, size_x, buffer_set->buffer_dim0_offset, temp_info, isTopRow, isBottomRow);
    unsigned char * cmpData = cmpkit_set->cmpData[next];
    unsigned char * qmean_pos = cmpData + FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + INT_BYTES * block_ind;
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks + cmpkit_set->offsets[next][x];
    unsigned char * prev_pos = encode_pos;
    const int * buffer_start_pos = buffer_set->currRow_data_pos;
    int * update_buffer_pos = buffer_set->updateRow_data_pos;
    for(int i=0; i<size_x; i++){
        for(size_t y=0; y<size.block_dim2; y++){
            int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
            const int * block_buffer_pos = buffer_start_pos;
            int * block_update_pos = update_buffer_pos;
            int pred = update_block_entries(block_buffer_pos, block_update_pos, buffer_set->buffer_dim0_offset, bias, block_size);
            unsigned char * sign_pos = cmpkit_set->signFlag;
            unsigned int * abs_err_pos = cmpkit_set->absPredError;
            int abs_err, max_err = 0;
            for(int j=0; j<block_size; j++){
                int err = (*block_update_pos++) - pred;
                *sign_pos++ = (err < 0);
                // std::cout << "iter "<<iter<<" block "<<block_ind<<" i = "<<i<<" j = "<<j<<": err = "<<err<<" sign = "<<+sign_pos[-1]<<" sign_pos = "<<sign_pos - cmpkit_set->signFlag << std::endl;
                abs_err = abs(err);
                *abs_err_pos++ = abs_err;
                max_err = max_err > abs_err ? max_err : abs_err;
            }
            buffer_start_pos += size.Bsize;
            update_buffer_pos += size.Bsize;
            int fixed_rate = max_err == 0 ? 0 : INT_BITS - __builtin_clz(max_err);
            cmpData[block_ind++] = (unsigned char)fixed_rate;
            for(int k=3; k>=0; k--){
                *(qmean_pos++) = (pred >> (8 * k)) & 0xff;
            }
            if(fixed_rate){
                unsigned int signbyteLength = convertIntArray2ByteArray_fast_1b_args(cmpkit_set->signFlag, block_size, encode_pos);
                encode_pos += signbyteLength;
                unsigned int savedbitsbyteLength = Jiajun_save_fixed_length_bits(cmpkit_set->absPredError, block_size, encode_pos, fixed_rate);
                encode_pos += savedbitsbyteLength;
            }
        }
        buffer_start_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
        update_buffer_pos += buffer_set->buffer_dim0_offset - size.block_dim2 * size.Bsize;
    }
    size_t increment = encode_pos - prev_pos;
    cmpkit_set->cmpSize += increment;
    cmpkit_set->prefix_length += increment;
    cmpkit_set->offsets[next][x+1] = cmpkit_set->prefix_length;
}

inline void heatdisUpdatePrePred(
    DSize_1d size,
    size_t numblockRow,
    SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_2d *buffer_set,
    TempInfo2D& temp_info,
    double errorBound, int current,
    int next, int iter
){
    size_t buffer_dim0_offset = buffer_set->buffer_dim0_offset;
    unsigned char * cmpData = cmpkit_set->cmpData[current];
    unsigned char * encode_pos = cmpData + (FIXED_RATE_PER_BLOCK_BYTES + INT_BYTES) * size.num_blocks;
    int * tempRow_pos = nullptr;
    cmpkit_set->reset();
    buffer_set->reset();
    extract_block_mean(cmpData+FIXED_RATE_PER_BLOCK_BYTES*size.num_blocks, cmpkit_set->mean_quant_inds, size.num_blocks);
    for(size_t x=0; x<numblockRow; x++){
        if(x == 0){
            recoverBlockRow2PrePred(x, size, cmpkit_set, encode_pos, buffer_set->currRow_data_pos, current, buffer_set->buffer_dim0_offset);
            recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, current, buffer_set->buffer_dim0_offset);
            heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, true, false);
        }else{
            rotate_buffer(buffer_set->currRow_data_pos, buffer_set->prevRow_data_pos, buffer_set->nextRow_data_pos, tempRow_pos);
            if(x == numblockRow - 1){
                heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, false, true);
            }else{
                recoverBlockRow2PrePred(x+1, size, cmpkit_set, encode_pos, buffer_set->nextRow_data_pos, current, buffer_set->buffer_dim0_offset);
                heatdisProcessCompressBlockRowPrePred(x, size, temp_info, cmpkit_set, buffer_set, errorBound, iter, next, false, false);
            }
        }
    }
}

template <class T>
inline void heatdisUpdatePrePred(
    DSize_1d& size,
    size_t numblockRow,
    SZxCmpBufferSet *cmpkit_set,
    SZxAppBufferSet_2d *buffer_set,
    TempInfo2D& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    T * h = (T *)malloc(size.nbEle * sizeof(T));
    int current = 0, next = 1;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        if(verb){
            if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
                SZx_decompress_1dMeanbased(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
                std::string h_name = heatdis2d_data_dir + "/h.M1.pre." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, size.nbEle);
                cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
                printf("prepred iter %d: cr = %.2f\n", iter-1, 1.0 * size.nbEle * sizeof(T) / cmpSize);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdisUpdatePrePred(size, numblockRow, cmpkit_set, buffer_set, temp_info, errorBound, current, next, iter);
        current = next;
        next = 1 - current;
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        SZx_decompress_1dMeanbased(h, cmpkit_set->cmpData[current], size.dim1, size.dim2, size.Bsize, errorBound);
        std::string h_name = heatdis2d_data_dir + "/h.M1.pre." + std::to_string(iter);
        writefile(h_name.c_str(), h, size.nbEle);
        cmpSize = FIXED_RATE_PER_BLOCK_BYTES * size.num_blocks + cmpkit_set->cmpSize;
        printf("prepred exit cr = %.2f\n", 1.0 * size.nbEle * sizeof(T) / cmpSize);
    }
    printf("prepred elapsed_time = %.6f\n", elapsed_time);
    free(h);
}

template <class T>
inline void heatdisUpdateDOC(
    DSize_1d& size,
    size_t dim1_padded,
    size_t dim2_padded,
    TempInfo2D& temp_info,
    double errorBound,
    int max_iter,
    bool verb
){
    struct timespec start, end;
    double elapsed_time = 0;
    size_t cmpSize;
    size_t nbEle_padded = dim1_padded * dim2_padded;
    T * h = (T *)malloc(nbEle_padded * sizeof(T));
    T * h2 = (T *)malloc(nbEle_padded * sizeof(T));
    unsigned char * compressed = (unsigned char *)malloc(nbEle_padded * sizeof(T));
    HeatDis2D heatdis(temp_info.src_temp, temp_info.wall_temp, temp_info.ratio, size.dim1, size.dim2);
    heatdis.initData(h, h2, temp_info.init_temp);
    SZx_compress_1dMeanbased(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
    T * tmp = nullptr;
    int iter = 0;
    while(iter < max_iter){
        iter++;
        clock_gettime(CLOCK_REALTIME, &start);
        SZx_decompress_1dMeanbased(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
        if(verb){
            if(iter >= ht2d_plot_offset && iter % ht2d_plot_gap == 0){
                std::string h_name = heatdis2d_data_dir + "/h.M1.doc." + std::to_string(iter-1);
                writefile(h_name.c_str(), h, nbEle_padded);
                printf("doc iter %d: cr = %.2f\n", iter-1, 1.0 * nbEle_padded * sizeof(T) / cmpSize);
            }
        }
        clock_gettime(CLOCK_REALTIME, &start);
        heatdis.reset_source(h, h2);
        heatdis.iterate(h, h2, tmp);
        SZx_compress_1dMeanbased(h, compressed, dim1_padded, dim2_padded, size.Bsize, errorBound, cmpSize);
        clock_gettime(CLOCK_REALTIME, &end);
        elapsed_time += get_elapsed_time(start, end);
    }
    {
        std::string h_name = heatdis2d_data_dir + "/h.M1.doc." + std::to_string(iter);
        writefile(h_name.c_str(), h, nbEle_padded);
        printf("doc exit cr = %.2f\n", 1.0 * nbEle_padded * sizeof(T) / cmpSize);
    }
    printf("doc elapsed_time = %.6f\n", elapsed_time);
    free(compressed);
    free(h);
    free(h2);
}

template <class T>
void SZx_heatdis_1dMeanbased(
    unsigned char *cmpDataBuffer,
    size_t dim1, size_t dim2,
    int blockSideLength, int max_iter,
    float source_temp, float wall_temp,
    float init_temp, float ratio,
    double errorBound,
    decmpState state, bool verb
){
    DSize_1d size(dim1, dim2, blockSideLength);
    size_t numblockRow = (size.dim1 - 1) / size.Bwidth + 1;
    size_t dim1_padded = size.dim1 + 2;
    size_t buffer_dim1 = size.Bwidth + 2;
    size_t buffer_dim2 = size.dim2 + 2;
    size_t buffer_size = buffer_dim1 * buffer_dim2;
    size_t nbEle_padded = dim1_padded * buffer_dim2;
    int * Buffer_2d = (int *)malloc(buffer_size * 4 * sizeof(int));
    unsigned int * absPredError = (unsigned int *)malloc(size.max_num_block_elements * sizeof(unsigned int));
    int * signPredError = (int *)malloc(size.max_num_block_elements * sizeof(int));
    unsigned char * signFlag = (unsigned char *)malloc(size.max_num_block_elements * sizeof(unsigned char));
    int * blocks_mean_quant = (int *)malloc(size.num_blocks * sizeof(int));

    unsigned char **cmpData = (unsigned char **)malloc(2 * sizeof(unsigned char *));
    int **offsets = (int **)malloc(2*sizeof(int *));
    for(int i=0; i<2; i++){
        cmpData[i] = (unsigned char *)malloc(nbEle_padded * sizeof(T)*2);
        offsets[i] = (int *)malloc(numblockRow * sizeof(int));
    }
    memcpy(cmpData[0], cmpDataBuffer, size.nbEle * sizeof(T));

    size_t prefix_length = 0;
    int block_index = 0;
    for(size_t x=0; x<numblockRow; x++){
        offsets[0][x] = prefix_length;
        offsets[1][x] = 0;
        int size_x = ((x+1) * size.Bwidth < size.dim1) ? size.Bwidth : size.dim1 - x*size.Bwidth;
        for(int i=0; i<size_x; i++){
            for(size_t y=0; y<size.block_dim2; y++){
                int block_size = ((y+1)*size.Bsize < size.dim2) ? size.Bsize : size.dim2 - y*size.Bsize;
                int cmp_block_sign_length = (block_size + 7) / 8;
                int fixed_rate = (int)cmpDataBuffer[block_index++];
                size_t savedbitsbytelength = compute_encoding_byteLength(block_size, fixed_rate);
                if(fixed_rate)
                    prefix_length += (cmp_block_sign_length + savedbitsbytelength);
            }
        }
    }
    TempInfo2D temp_info(source_temp, wall_temp, init_temp, ratio, errorBound);
    SZxAppBufferSet_2d * buffer_set = new SZxAppBufferSet_2d(buffer_dim1, buffer_dim2, Buffer_2d, appType::HEATDIS);
    SZxCmpBufferSet * cmpkit_set = new SZxCmpBufferSet(cmpData, offsets, blocks_mean_quant, absPredError, signPredError, signFlag);

    switch(state){
        case decmpState::postPred:{
            break;
        }
        case decmpState::prePred:{
            heatdisUpdatePrePred<T>(size, numblockRow, cmpkit_set, buffer_set, temp_info, errorBound, max_iter, verb);
            break;
        }
        case decmpState::full:{
            heatdisUpdateDOC<T>(size, dim1_padded, buffer_dim2, temp_info, errorBound, max_iter, verb);
            break;
        }
    }

    delete buffer_set;
    delete cmpkit_set;
    free(Buffer_2d);
    free(absPredError);
    free(signPredError);
    free(signFlag);
    free(blocks_mean_quant);
    for(int i=0; i<2; i++){
        free(cmpData[i]);
        free(offsets[i]);
    }
    free(cmpData);
    free(offsets);
}

template <class T>
void SZx_heatdis_1dMeanbased(
    unsigned char *cmpDataBuffer, ht2DSettings& s, decmpState state, bool verb
){
    SZx_heatdis_1dMeanbased<T>(cmpDataBuffer, s.dim1, s.dim2, s.B, s.steps, s.src_temp, s.wall_temp, s.init_temp, s.ratio, s.eb, state, verb);
}

#endif