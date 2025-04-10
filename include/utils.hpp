#ifndef _COMP_UTILS_HPP
#define _COMP_UTILS_HPP

#include <cstdint>
#include <iostream>
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

template<class T>
void set_relative_eb(const std::vector<T>& oriData_vec, double& errorBound){
    auto max_val = *std::max_element(oriData_vec.begin(), oriData_vec.end());
    auto min_val = *std::min_element(oriData_vec.begin(), oriData_vec.end());
    auto range = max_val - min_val;
    errorBound *= range;
    printf("Max = %.4e, min = %.4e, range = %.6e, abs_eb = %.6e\n", max_val, min_val, range, errorBound);
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
    int pos = 0, n = 0;
    double max_error = 0;
    double v1, v2;
    for(size_t i=1; i<dim1-1; i++){
        const T * ori_pos = oriData + i * dim2;
        const T * dec_pos = decData + i * dim2;
        for(size_t j=1; j<dim2-1; j++){
            double diff = fabs(ori_pos[j] - dec_pos[j]);
            if(diff > max_error){
                max_error = diff;
                pos = n;
                v1 = ori_pos[j];
                v2 = dec_pos[j];
            }
            n++;
        }
    }
    // std::cout << "max_error = " << max_error << ", pos = " << pos << ", x = " << pos / (dim2-2) + 1 << ", y = " << pos % (dim2-2) + 1 << std::endl;
    // std::cout << "ori = " << v1 << ", dec = " << v2 << std::endl;
    return max_error;
}

template <class T>
double verify_dxdydz(const T *oriData, const T *decData, size_t dim1, size_t dim2, size_t dim3)
{
    size_t dim0_offset = dim2 * dim3;
    size_t dim1_offset = dim3;
    int pos = 0, n = 0;
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
                    pos = n;
                    v1 = y_ori_pos[k];
                    v2 = y_dec_pos[k];
                }
                n++;
            }
        }
    }
    // std::cout << "max_error = " << max_error << ", pos = " << pos << ", x = " << pos / ((dim2-2) * (dim3-2)) + 1 << ", y = " << (pos % ((dim2-2) * (dim3-2))) / (dim3-2) + 1 << ", z = " << pos % (dim3-2) + 1 << std::endl;
    // std::cout << "ori = " << v1 << ", dec = " << v2 << std::endl;
    return max_error;
}

template <class T>
void print_matrix_float(int dim1, int dim2, std::string name, T *mat)
{
    std::cout << "--------- " << name << " ---------" << std::endl;
    for(int i=0; i<dim1; i++){
        for(int j=0; j<dim2; j++){
            printf("%.4f  ", mat[i*dim2+j]);
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

class HeatDis2D{
public:
	size_t dim1, dim2;
	size_t dim1_padded, dim2_padded;
	size_t nbEle_padded;
	size_t c1, c2;
	float source_temp;
	float wall_temp;
	float ratio;
	HeatDis2D(float src, float wall, float r, size_t d1, size_t d2)
		: source_temp(src), wall_temp(wall), ratio(r), dim1(d1), dim2(d2){
        dim1_padded = dim1 + 2;
        dim2_padded = dim2 + 2;
		nbEle_padded = dim1_padded * dim2_padded;
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
		memset(h, 0, nbEle_padded * sizeof(T));
		T * g = h + 1;
		for(int j=0; j<dim2; j++){
			g[j] = wall_temp;
			if(j>=c1 && j<=c2) g[j] = source_temp;				
		}
		g = h + (dim1_padded - 1) * dim2_padded;
		for(int j=0; j<dim2_padded; j++){
			g[j] = wall_temp;
		}
		for(int i=0; i<dim1_padded; i++){
			h[i * dim2_padded] = wall_temp, h[(i + 1) * dim2_padded - 1] = wall_temp;
		}
		for(int i=1; i<=dim1; i++) {
			for (int j=1; j<=dim2; j++) {
				h[i * dim2_padded + j] = init_temp;
			}
		}
		memcpy(h2, h, dim1_padded * dim2_padded * sizeof(T));
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
		for(int i=1; i<=dim1; i++){
			for(int j=1; j<=dim2; j++){
				h2[i*dim2_padded+j] = 0.25 * (h[(i*dim2_padded)+j-1] + h[(i*dim2_padded)+j+1] + h[(i-1)*dim2_padded+j] + h[(i+1)*dim2_padded+j]);
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
	void doWork(T*& h, T*& h2, double criteria, int num_iter, int plot_gap, int plot_offset=1){
		T * tmp = nullptr;
        int iter = 0;
        while(iter < num_iter){
            iter++;
			if(iter >= plot_offset && iter % plot_gap == 0){
				std::string h_name = heatdis2d_data_dir + "/h.ref." + std::to_string(iter-1);
				writefile(h_name.c_str(), h, nbEle_padded);
			}
			iterate(h, h2, tmp);
        }
        {
            std::string h_name = heatdis2d_data_dir + "/h.ref." + std::to_string(iter);
            writefile(h_name.c_str(), h, nbEle_padded);
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
    const double ub = 1.0;
    const double vb = 0.0;
	int UBorderVal;
	int VBorderVal;
	const int d = 6;
    size_t L;
    size_t L_padded;
    size_t nbEle;
	size_t nbEle_padded;
    size_t dim0_offset;
    size_t dim1_offset;
    double Du, Dv, F, k, Fk, dt;
    double Dudt, Dvdt, ebdt, Fdt_fl, Fkdt;
    int64_t Fdt_int;
    GrayScott(size_t l, double Du_, double Dv_, double dt_, double Feed, double kill, double eb)
        : L(l), Du(Du_), Dv(Dv_), dt(dt_), F(Feed), k(kill){
        L_padded = L + 2;
		nbEle = L * L * L;
		nbEle_padded = L_padded * L_padded * L_padded;
        dim0_offset = L_padded * L_padded;
        dim1_offset = L_padded;
        UBorderVal = (int64_t)SZ_quantize(ub, eb);
        VBorderVal = (int64_t)SZ_quantize(vb, eb);
        Fdt_int = (int64_t)SZ_quantize(F * dt, eb);
		Dudt = Du * dt / 6.0;
        Dvdt = Dv * dt / 6.0;
        ebdt = dt * 4 * eb * eb;
        Fdt_fl = F * dt;
		Fkdt = (F + k) * dt;
        Fk = F + k;
    }
    inline int c2i_noghost(int i, int j, int k){
        return i * L * L + j * L + k;
    }
    template <class T>
	void initData_noghost(T *u, T *v, T *u2, T *v2){
		for(int i=0; i<nbEle; i++){
			u[i] = ub, u2[i] = ub;
			v[i] = vb, v2[i] = vb;
		}
		const int le = L / 2 - d;
		const int ue = L / 2 + d;
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
			u[i] = ub, u2[i] = ub;
			v[i] = vb, v2[i] = vb;
		}
		const int le = L / 2 - d + 1;
		const int ue = L / 2 + d + 1;
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
	void doWork(T*& u, T*& v, T*& u2, T*& v2, double criteria, int num_iter, int plot_gap, int plot_offset=1){
		T * tmp = nullptr;
        int iter = 0;
        while(iter < num_iter){
            iter++;
			if(iter >= plot_offset && iter % plot_gap == 0){
				std::string u_name = grayscott_data_dir + "/u.ref." + std::to_string(iter-1);
				std::string v_name = grayscott_data_dir + "/v.ref." + std::to_string(iter-1);
				writefile(u_name.c_str(), u, nbEle_padded);
				writefile(v_name.c_str(), v, nbEle_padded);
			}
			iterate(u, v, u2, v2, tmp);
        }
        {
            std::string u_name = grayscott_data_dir + "/u.ref." + std::to_string(iter);
            std::string v_name = grayscott_data_dir + "/v.ref." + std::to_string(iter);
            writefile(u_name.c_str(), u, nbEle_padded);
            writefile(v_name.c_str(), v, nbEle_padded);
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

class HeatDis3D{
public:
	size_t dim1, dim2, dim3;
    size_t dim0_offset, dim1_offset;
	size_t dim1_padded, dim2_padded, dim3_padded;
	size_t nbEle_padded;
    size_t dim0_offset_padded, dim1_offset_padded;
	float T_top, T_bott;
	float T_wall;
    float alpha;
	HeatDis3D(float T_t, float T_b, float T_w, float al, size_t d1, size_t d2, size_t d3)
		: T_top(T_t), T_bott(T_b), T_wall(T_w), alpha(al), dim1(d1), dim2(d2), dim3(d3){
        assert(dim1 >= dim2);
        assert(dim1 >= dim3);
        dim0_offset = dim2 * dim3;
        dim1_offset = dim3;
        dim1_padded = dim1 + 2;
        dim2_padded = dim2 + 2;
        dim3_padded = dim3 + 2;
        dim0_offset_padded = dim2_padded * dim3_padded;
        dim1_offset_padded = dim3_padded;
		nbEle_padded = dim1_padded * dim2_padded * dim3_padded;
	}
	template <class T>
	void initData_noghost(T *h, T *h2, float T_init){
		for(int i=0; i<dim1; i++){
			for(int j=0; j<dim2; j++){
                for(int k=0; k<dim3; k++){
                    h[i * dim0_offset + j * dim1_offset + k] = T_init;
                    h2[i * dim0_offset + j * dim1_offset + k] = T_init;
                }
			}
		}
	}
    template <class T>
	void initData(T *h, T *h2, float T_init){
        int i, j, k;
        for(i=0; i<nbEle_padded; i++){
            h[i] = T_wall;
        }
        T * g1 = h + dim1_offset_padded + 1;
        T * g2 = h + (dim1 + 1) * dim0_offset_padded + dim1_offset_padded + 1;
        for(j=0; j<dim2; j++){
            for(k=0; k<dim3; k++){
                g1[k] = T_top;
                g2[k] = T_bott;
            }
            g1 += dim1_offset_padded;
            g2 += dim1_offset_padded;
        }
		for(i=1; i<=dim1; i++){
			for(j=1; j<=dim2; j++){
                for(k=1; k<=dim3; k++){
                    h[i * dim0_offset_padded + j * dim1_offset_padded + k] = T_init;
                }
			}
		}
		memcpy(h2, h, nbEle_padded * sizeof(T));
	}
	template <class T>
	void reset_source(T *h){
        T * g1 = h + dim1_offset_padded + 1;
        T * g2 = h + (dim1 + 1) * dim0_offset_padded + dim1_offset_padded + 1;
        for(int j=0; j<dim2; j++){
            for(int k=0; k<dim3; k++){
                g1[k] = T_top;
                g2[k] = T_bott;
            }
            g1 += dim1_offset_padded;
            g2 += dim1_offset_padded;
        }
	}
    inline int c2i(int i, int j, int k){
        return i * dim0_offset_padded + j * dim1_offset_padded + k;
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
        return ts;
    }
    template <class T>
	void calc(const T *h, T *h2){
		for(int i=1; i<=dim1; i++){
			for(int j=1; j<=dim2; j++){
                for(int k=1; k<=dim3; k++){
                    int index = c2i(i, j, k);
                    h2[index] = h[index] + alpha * laplacian(i, j, k, h);
                }
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
	void doWork(T*& h, T*& h2, double criteria, int num_iter, int plot_gap, int plot_offset=1){
		T * tmp = nullptr;
        int iter = 0;
        while(iter < num_iter){
            iter++;
			if(iter >= plot_offset && iter % plot_gap == 0){
				std::string h_name = heatdis3d_data_dir + "/h.ref." + std::to_string(iter-1);
				writefile(h_name.c_str(), h, nbEle_padded);
			}
			iterate(h, h2, tmp);
        }
        {
            std::string h_name = heatdis3d_data_dir + "/h.ref." + std::to_string(iter);
            writefile(h_name.c_str(), h, nbEle_padded);
        }
	}
    template <class T>
	void trimData(T *padded, T *trimmed){
		for(int i=0; i<dim1; i++){
			T * raw = padded + (i + 1) * dim0_offset_padded + dim1_offset_padded + 1;
			T * data = trimmed + i * dim0_offset;
			for(int j=0; j<dim2; j++){
				memcpy(data, raw, dim3 * sizeof(T));
				raw += dim1_offset_padded;
				data += dim3;
			}
		}
	}
};

#endif