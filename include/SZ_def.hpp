#ifndef _SZ_DEF_HPP
#define _SZ_DEF_HPP

#include <cstdlib>
#include <cstring>

#define INT_BITS 32
#define INT_BYTES 4
#define FLOAT_BYTES 4
#define FIXED_RATE_PER_BLOCK_BYTES 1
#define REG_COEFF_SIZE_2D 3
#define REG_COEFF_SIZE_3D 4

std::string data_file_2d = "/pscratch/xli281_uksr/xwu/datasets/CESM/CLDHGH_1_1800_3600.f32";
std::string data_file_3d = "/pscratch/xli281_uksr/xwu/datasets/Hurricane/data/VelocityX.f32";

enum class appType
{
    MEAN,
    CENTRALDIFF,
    HEATDIS
};

enum class decmpState
{
	full,
	prePred,
    postPred
};

decmpState intToDecmpState(int value){
    switch(value){
        case 0: return decmpState::full;
        case 1: return decmpState::prePred;
        case 2: return decmpState::postPred;
        default: 
            throw std::invalid_argument("Invalid integer for decmpState");
    }
}

struct DSize_1d
{
	size_t dim1;
	size_t dim2;
	size_t nbEle;
	int Bsize;
	int Bwidth;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t num_blocks;
	size_t dim0_offset;
	DSize_1d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		Bwidth = (int)std::sqrt(bs);
		max_num_block_elements = bs;
		block_dim1 = r1;
		block_dim2 = (r2 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2;
		dim0_offset = r2;
	}
};

struct DSize_2d
{
	size_t dim1;
	size_t dim2;
	size_t nbEle;
	int Bsize;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t num_blocks;
	size_t dim0_offset;
	DSize_2d(size_t r1, size_t r2, int bs){
		dim1 = r1, dim2 = r2;
		nbEle = r1 * r2;
		Bsize = bs;
		max_num_block_elements = bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2;
		dim0_offset = r2;
	}
};

struct DSize_3d
{
	size_t dim1;
	size_t dim2;
	size_t dim3;
	size_t nbEle;
	int Bsize;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t block_dim3;
	size_t num_blocks;
	size_t dim0_offset;
	size_t dim1_offset;
	DSize_3d(size_t r1, size_t r2, size_t r3, int bs){
		dim1 = r1, dim2 = r2, dim3 = r3;
		nbEle = r1 * r2 * r3;
		Bsize = bs;
		max_num_block_elements = bs * bs * bs;
		block_dim1 = (r1 - 1) / bs + 1;
		block_dim2 = (r2 - 1) / bs + 1;
		block_dim3 = (r3 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2 * block_dim3;
		dim0_offset = r2 * r3;
		dim1_offset = r3;
	}
};

struct DSize_3d2d
{
	size_t dim1;
	size_t dim2;
	size_t dim3;
	size_t nbEle;
	int Bsize;
	int Bwidth;
	int max_num_block_elements;
	size_t block_dim1;
	size_t block_dim2;
	size_t block_dim3;
	size_t num_blocks;
	size_t dim0_offset;
	size_t dim1_offset;
	DSize_3d2d(size_t r1, size_t r2, size_t r3, int bs){
		dim1 = r1, dim2 = r2, dim3 = r3;
		nbEle = r1 * r2 * r3;
		Bsize = bs;
		Bwidth = bs;
		max_num_block_elements = bs * bs;
		block_dim1 = r1;
		block_dim2 = (r2 - 1) / bs + 1;
		block_dim3 = (r3 - 1) / bs + 1;
		num_blocks = block_dim1 * block_dim2 * block_dim3;
		dim0_offset = r2 * r3;
		dim1_offset = r3;
	}
};

template <class T>
inline int SZ_quantize(const T& data, double errorBound)
{
    return static_cast<int>(std::floor((data + errorBound) / (2 * errorBound)));
}

template <class T>
inline void exchange_buffer(
    T *& curr, T *& next, T *& temp
){
    temp = curr;
    curr = next;
    next = temp;
}

template <class T>
inline void rotate_buffer(
    T *& curr, T *& prev, T *& next, T *& temp
){
    temp = prev;
    prev = curr;
    curr = next;
    next = temp;
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
	void doWork(T *h_, T *h2_, int num_iter){
		T * h = h_, * h2 = h2_;
		T * tmp = nullptr;
		for(int iter=0; iter<num_iter; iter++){
			iterate(h, h2, tmp);
		}
	}
    template <class T>
	void doWork(T *h_, T *h2_, int num_iter, int plotgap){
		T * h = h_, * h2 = h2_;
		T * tmp = nullptr;
		for(int iter=0; iter<num_iter; iter++){
			iterate(h, h2, tmp);
			if(iter % plotgap == 0){
				std::string h_name = "h.dat." + std::to_string(iter);
				writefile(h_name.c_str(), h, nbEle_padded);
				std::cout << "saved snapshot " << iter << std::endl;
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
		int UBorderVal, UBorderVal2, VBorderVal;
        int dt, Dudt, Dvdt, Fdt, Fkdt;
        gsIntCoeff(double dt_, double dudt, double dvdt,
		double fdt, double fkdt, double eb, double ub, double ub2, double vb){
            dt = SZ_quantize(dt_, eb);
            Dudt = SZ_quantize(dudt, eb);
            Dvdt = SZ_quantize(dvdt, eb);
            Fdt = SZ_quantize(fdt, eb);
            Fkdt = SZ_quantize(fkdt, eb);
			UBorderVal = SZ_quantize(ub, eb);
			UBorderVal2 = SZ_quantize(ub2, eb);
			VBorderVal = SZ_quantize(vb, eb);
        }
    };
	const double UBorderVal = 1.0;
	const double UBorderVal2 = 0.0;
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
		Dudt = Du * dt / 6.0, Dvdt = Dv * dt / 6.0;
		Fkdt = (F + k) * dt, Fdt = F * dt;
        Fk = F + k;
		intCoeff = new gsIntCoeff(dt, Dudt, Dvdt, Fdt, Fkdt, eb, UBorderVal, UBorderVal2, VBorderVal);
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
			u[i] = UBorderVal, u2[i] = UBorderVal2;
			v[i] = VBorderVal, v2[i] = VBorderVal;
		}
		const int le = L / 2 - d - 1;
		const int ue = L / 2 + d - 1;
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
			u[i] = UBorderVal, u2[i] = UBorderVal2;
			v[i] = VBorderVal, v2[i] = VBorderVal;
		}
		const int le = L / 2 - d;
		const int ue = L / 2 + d;
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
	void doWork(T *u_, T *v_, T *u2_, T *v2_, int num_iter){
		T * u = u_, * v = v_;
		T * u2 = u_, * v2 = v2_;
		T * tmp = nullptr;
		for(int iter=0; iter<num_iter; iter++){
			iterate(u, v, u2, v2, tmp);
		}
	}
    template <class T>
	void doWork(T *u_, T *v_, T *u2_, T *v2_, int num_iter, int plotgap){
		T * u = u_, * v = v_;
		T * u2 = u_, * v2 = v2_;
		T * tmp = nullptr;
		for(int iter=1; iter<=num_iter; iter++){
			iterate(u, v, u2, v2, tmp);
			if(iter % plotgap == 0){
				std::string u_name = "U.dat." + std::to_string(iter);
				std::string v_name = "V.dat." + std::to_string(iter);
				writefile(u_name.c_str(), u, nbEle_padded);
				writefile(v_name.c_str(), u, nbEle_padded);
				std::cout << "saved snapshot " << iter << std::endl;
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
