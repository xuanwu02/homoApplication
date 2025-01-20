#ifndef _COMP_UTILS_HPP
#define _COMP_UTILS_HPP

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <fstream>
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

template<class T>
void set_relative_eb(const std::vector<T>& oriData_vec, double& errorBound){
    auto max_val = *std::max_element(oriData_vec.begin(), oriData_vec.end());
    auto min_val = *std::min_element(oriData_vec.begin(), oriData_vec.end());
    auto range = max_val - min_val;
    errorBound *= range;
    std::cout << "Max = " << max_val << ", min = " << min_val << ", range = " << range << ", abs_eb = " << errorBound << std::endl;
}

double get_elapsed_time(struct timespec &start, struct timespec &end){
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;
}

// template <class T>
// void initRandomData(T min, T max, unsigned int seed, size_t n, T *data){
//     std::mt19937 generator(seed);  
//     std::uniform_real_distribution<T> distribution(min, max);
//     for(size_t i=0; i<n; i++){
//         data[i] = distribution(generator);
//     }
// }

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
void initData(
    size_t dim1, size_t dim2,
    T *h, T init_temp
){
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            h[i * dim2 + j] = init_temp;
        }
    }
}
template <class T>
void doWork(
    size_t dim1, size_t dim2, T *g, T *h,
    T src_temp, T wall_temp, double ratio
){
    memcpy(h, g, dim1 * dim2 * sizeof(float));
    size_t c1 =  dim2 * (1.0 - ratio) * 0.5 + 1;
    size_t c2 =  dim2 * (1.0 + ratio) * 0.5 - 1;
    T left, right, top, bottom;
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            size_t index = i * dim2 + j;
            bool j_flag = (j >= c1) && (j <= c2);
            left = (j == 0) ? wall_temp : h[index - 1];
            right = (j == dim2 - 1) ? wall_temp : h[index + 1];
            top = (i > 0) ? h[index - dim2] : (j_flag ? src_temp : wall_temp);
            bottom = (i == dim1 - 1) ? wall_temp : h[index + dim2];
            g[index] = 0.25 * (left + right + top + bottom);
        }
    }
}
template <class T>
void doWork(
    size_t dim1, size_t dim2, int max_iter, T *g, T *h,
    T src_temp, T wall_temp, double ratio
){
    for(int i=0; i<max_iter; i++){
        doWork(dim1, dim2, g, h, src_temp, wall_temp, ratio);
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
    for(i=0; i<dim1; i++){
        curr_row = data + i * dim2;
        prev_row = i == 0 ? curr_row : curr_row - dim2;
        next_row = i == dim1-1 ? curr_row : curr_row + dim2;
        dx_pos = dx_result + i * dim2;
        dy_pos = dy_result + i * dim2;
        T coeff_dx = (i == 0 || i == dim1 - 1) ? 1.0 : 0.5;
        for(j=0; j<dim2; j++){
            dx_pos[j] = (next_row[j] - prev_row[j]) * coeff_dx;
            size_t prev_j = j == 0 ? j : j - 1;
            size_t next_j = j == dim2 - 1 ? j : j + 1;
            T coeff_dy = (j == 0) || (j == dim2 - 1) ? 1.0 : 0.5;
            dy_pos[j] = (curr_row[next_j] - curr_row[prev_j]) * coeff_dy;
        }
    }
}

template <class T>
void compute_dxdydz(
    size_t dim1, size_t dim2, size_t dim3, const T *data,
    T *dx_result, T *dy_result, T *dz_result
){
    size_t dim0_offset = dim2 * dim3;
    size_t dim1_offset = dim3;
    size_t i, j, k;
    const T * curr_plane = data;
    for(i=0; i<dim1; i++){
        const T * prev_plane = i == 0 ? curr_plane : curr_plane - dim0_offset;
        const T * next_plane = i == dim1 - 1 ? curr_plane : curr_plane + dim0_offset;
        const T * curr_row = curr_plane;
        T coeff_dx = (i == 0) || (i == dim1 - 1) ? 1.0 : 0.5;
        for(j=0; j<dim2; j++){
            const T * prev_row = j == 0 ? curr_row : curr_row - dim1_offset;
            const T * next_row = j == dim2 - 1 ? curr_row : curr_row + dim1_offset;
            T coeff_dy = (j == 0) || (j == dim2 - 1) ? 1.0 : 0.5;
            for(k=0; k<dim3; k++){
                size_t index = j * dim1_offset + k;
                size_t res_index = i * dim0_offset + index;
                size_t prev_k = k == 0 ? k : k - 1;
                size_t next_k = k == dim3 - 1 ? k : k + 1;
                T coeff_dz = (k == 0) || (k == dim3 - 1) ? 1.0 : 0.5;
                dx_result[res_index] = (next_plane[index] - prev_plane[index]) * coeff_dx;
                dy_result[res_index] = (next_row[k] - prev_row[k]) * coeff_dy;
                dz_result[res_index] = (curr_row[next_k] - curr_row[prev_k]) * coeff_dz;
            }
            curr_row += dim1_offset;
        }
        curr_plane += dim0_offset;
    }
}

class HeatDis{
private:
	size_t dim1, dim2;
	size_t nbEle_padded;
	size_t c1, c2;
	float source_temp;
	float wall_temp;
	float ratio;
public:
	HeatDis(float src, float wall, float r, size_t d1, size_t d2)
		: source_temp(src), wall_temp(wall), ratio(r), dim1(d1), dim2(d2){
		nbEle_padded = (dim1 + 2) * (dim2 + 2);
		c1 =  dim2 * (1.0 - ratio) * 0.5 + 1;
		c2 =  dim2 * (1.0 + ratio) * 0.5 - 1;
	}
	template <class T>
	void initData_noghost(T *h, T *h2, float init_temp){
		for(int i=0; i<dim1; i++){
			for(int j=0; j<dim2; j++){
				h[i * dim2 + j] = init_temp;
				h2[i * dim2 + j] = init_temp;
			}
		}
	}
    template <class T>
	void initData(T *h, T *h2, float init_temp){
		int buffer_dim1 = dim1 + 2;
		int buffer_dim2 = dim2 + 2;
		memset(h, 0, buffer_dim1 * buffer_dim2 * sizeof(T));
		T * g = h + 1;
		for(int j=0; j<dim2; j++){
			g[j] = wall_temp;
			if(j>=c1 && j<=c2) g[j] = source_temp;				
		}
		g = h + (buffer_dim1 - 1) * buffer_dim2;
		for(int j=0; j<buffer_dim2; j++){
			g[j] = wall_temp;
		}
		for(int i=0; i<buffer_dim1; i++){
			h[i * buffer_dim2] = wall_temp, h[(i + 1) * buffer_dim2 - 1] = wall_temp;
		}
		for(int i=1; i<=dim1; i++) {
			for (int j=1; j<=dim2; j++) {
				h[i * buffer_dim2 + j] = init_temp;
			}
		}
		memcpy(h2, h, buffer_dim1 * buffer_dim2 * sizeof(T));
	}
	template <class T>
	void reset_source(T *h, T *h2){
		T * g = h + 1, * g2 = h2 + 1;
		for(int j=0; j<dim2; j++){
			g[j] = wall_temp;
			g2[j] = wall_temp;
			if(j>=c1 && j<=c2){
				g[j] = source_temp;				
				g2[j] = source_temp;	
			}			
		}
	}
    template <class T>
	void calc(const T *h, T *h2){
		int M = dim2 + 2;
		for(int i=1; i<=dim1; i++){
			for(int j=1; j<=dim2; j++){
				h2[i*M+j] = 0.25 * (h[(i*M)+j-1] + h[(i*M)+j+1] + h[(i-1)*M+j] + h[(i+1)*M+j]);
			}
		}
	}
    template <class T>
	void iterate(T*& h, T*& h2, T*& tmp){
		calc(h, h2);
		exchange_buffer(h, h2, tmp);
	}
    template <class T>
	void doWork(T*& h, T*& h2, int num_iter){
		T * tmp = nullptr;
		for(int iter=1; iter<=num_iter; iter++){
			iterate(h, h2, tmp);
		}
	}
    template <class T>
	void doWork(T*& h, T*& h2, int num_iter, int plotgap){
		T * tmp = nullptr;
		for(int iter=1; iter<=num_iter; iter++){
			iterate(h, h2, tmp);
			if(iter % plotgap == 0){
				std::string h_name = "/Users/xuanwu/github/backup/homoApplication/plot/ht_data/dec/h.ref." + std::to_string(iter);
				writefile(h_name.c_str(), h, nbEle_padded);
			}
		}
	}
    template <class T>
	void trimData(T *padded, T *trimmed){
		int buffer_dim0_offset = dim2 + 2;
		T * raw = padded + buffer_dim0_offset + 1;
		T * data = trimmed;
		for(int i=0; i<dim1; i++){
			memcpy(data, raw, dim2 * sizeof(T));
			raw += buffer_dim0_offset;
			data += dim2;
		}
	}
};

class GrayScott{
public:
    class gsIntCoeff{
    public:
		int64_t UBorderVal, VBorderVal;
        int64_t dt, Dudt, Dvdt, Fdt, Fkdt;
        gsIntCoeff(double dt_, double dudt, double dvdt,
		double fdt, double fkdt, double eb, double ub, double vb){
            dt = (int64_t)SZ_quantize(dt_, eb);
            Dudt = (int64_t)SZ_quantize(dudt, eb);
            Dvdt = (int64_t)SZ_quantize(dvdt, eb);
            Fdt = (int64_t)SZ_quantize(fdt, eb);
            Fkdt = (int64_t)SZ_quantize(fkdt, eb);
			UBorderVal = (int64_t)SZ_quantize(ub, eb);
			VBorderVal = (int64_t)SZ_quantize(vb, eb);
        }
    };
	const double UBorderVal = 1.0;
	const double VBorderVal = 0.0;
	const int d = 6;
    size_t L;
    size_t nbEle;
	size_t nbEle_padded;
    size_t dim0_offset;
    size_t dim1_offset;
    double Du, Dv, F, k, dt;
    double Fk, Dudt, Dvdt, Fdt, Fkdt;
	gsIntCoeff *intCoeff;
    GrayScott(size_t l, double Du_, double Dv_, double dt_, double Feed, double kill, double eb)
        : L(l), Du(Du_), Dv(Dv_), dt(dt_), F(Feed), k(kill){
		nbEle = L * L * L;
		nbEle_padded = (L + 2) * (L + 2) * (L + 2);
        dim0_offset = (L + 2) * (L + 2);
        dim1_offset = (L + 2);
		Dudt = Du * dt, Dvdt = Dv * dt;
		// Dudt = Du * dt / 6.0, Dvdt = Dv * dt / 6.0;
		Fkdt = (F + k) * dt, Fdt = F * dt;
        Fk = F + k;
		intCoeff = new gsIntCoeff(dt, Dudt, Dvdt, Fdt, Fkdt, eb, UBorderVal, VBorderVal);
    }
    ~GrayScott(){
        delete intCoeff;
    }
    inline int c2i_noghost(int i, int j, int k){
        return i * L * L + j * L + k;
    }
    template <class T>
	void initData_noghost(T *u, T *v, T *u2, T *v2){
		for(int i=0; i<nbEle; i++){
			u[i] = UBorderVal, u2[i] = UBorderVal;
			v[i] = VBorderVal, v2[i] = VBorderVal;
		}
		const int le = L / 2 - d - 1;
		const int ue = L / 2 + d - 1;
		// const int le = 2;
		// const int ue = 4;
		for(int i=le; i<ue; i++){
			for(int j=le; j<ue; j++){
				for(int k=le; k<ue; k++){
					int index = c2i_noghost(i, j, k);
					u[index] = 0.25;
					v[index] = 0.33;
				}
			}
		}
	}
    inline int c2i(int i, int j, int k){
        return i * dim0_offset + j * dim1_offset + k;
    }
    template <class T>
	void initData(T *u, T *v, T *u2, T *v2){
		for(int i=0; i<nbEle_padded; i++){
			u[i] = UBorderVal, u2[i] = UBorderVal;
			v[i] = VBorderVal, v2[i] = VBorderVal;
		}
		const int le = L / 2 - d;
		const int ue = L / 2 + d;
		// const int le = 3;
		// const int ue = 5;
		for(int i=le; i<ue; i++){
			for(int j=le; j<ue; j++){
				for(int k=le; k<ue; k++){
					int index = c2i(i, j, k);
					u[index] = 0.25;
					v[index] = 0.33;
				}
			}
		}
	}
    template <class T>
    inline T calcU(T tu, T tv){
        return -tu * tv * tv + F * (1.0 - tu);
    }
    template <class T>
    inline T calcV(T tu, T tv){
        return tu * tv * tv - Fk * tv;
    }
    template <class T>
    inline T laplacian(int i, int j, int k, const T *s){
        T ts = 0.0;
        ts += s[c2i(i - 1, j, k)];
        ts += s[c2i(i + 1, j, k)];
        ts += s[c2i(i, j - 1, k)];
        ts += s[c2i(i, j + 1, k)];
        ts += s[c2i(i, j, k - 1)];
        ts += s[c2i(i, j, k + 1)];
        ts += -6.0 * s[c2i(i, j, k)];
        return ts / 6.0;
    }
    template <class T>
	void calc(const T *u, const T *v, T *u2, T *v2){
		for(int i=1; i<=L; i++){
			for(int j=1; j<=L; j++){
				for(int k=1; k<=L; k++){
					const int index = c2i(i, j, k);
					T du = Du * laplacian(i, j, k, u) + calcU(u[index], v[index]);
					T dv = Dv * laplacian(i, j, k, v) + calcV(u[index], v[index]);
					u2[index] = u[index] + du * dt;
					v2[index] = v[index] + dv * dt;
				}
			}
		}
	}
    template <class T>
	void iterate(T*& u, T*& v, T*& u2, T*& v2, T*& tmp){
		calc(u, v, u2, v2);
		exchange_buffer(u, u2, tmp);
		exchange_buffer(v, v2, tmp);
	}
    template <class T>
	void doWork(T*& u, T*& v, T*& u2, T*& v2, int num_iter){
		T * tmp = nullptr;
		for(int iter=1; iter<=num_iter; iter++){
			iterate(u, v, u2, v2, tmp);
		}
	}
    template <class T>
	void doWork(T*& u, T*& v, T*& u2, T*& v2, int num_iter, int plotgap){
		T * tmp = nullptr;
		for(int iter=1; iter<=num_iter; iter++){
			iterate(u, v, u2, v2, tmp);
			if(iter % plotgap == 0){
				std::string u_name = "/Users/xuanwu/github/backup/homoApplication/plot/gs_data/u.ref." + std::to_string(iter);
				std::string v_name = "/Users/xuanwu/github/backup/homoApplication/plot/gs_data/v.ref." + std::to_string(iter);
				writefile(u_name.c_str(), u, nbEle_padded);
				writefile(v_name.c_str(), v, nbEle_padded);
			}
		}
	}
    template <class T>
	void trimData(T *padded, T *trimmed){
		int data_dim0_offset = L * L;
		for(int i=0; i<L; i++){
			T * raw = padded + (i + 1) * dim0_offset + dim1_offset + 1;
			T * data = trimmed + i * data_dim0_offset;
			for(int j=0; j<L; j++){
				memcpy(data, raw, L * sizeof(T));
				raw += dim1_offset;
				data += L;
			}
		}
	}
};

#endif