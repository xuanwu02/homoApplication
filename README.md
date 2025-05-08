# Installation
mkdir build && cd build <br>
cmake .. <br>
make <br>

# Example data dimensions
ocean_uv.f32 (2D): 2400 x 3600 <br>
hurricane_uv.f32 (3D): 100 x 500 x 500 <br>

# Run tests
## Mean
./build/test/mean_szp_1d ./data/ocean_uv.f32 2 2400 3600 1 256 1e-1 0 <br>
./build/test/mean_szp_1d ./data/ocean_uv.f32 2 2400 3600 1 256 1e-1 1 <br>
./build/test/mean_szp_1d ./data/ocean_uv.f32 2 2400 3600 1 256 1e-1 2 <br>

./build/test/mean_szp_2d ./data/ocean_uv.f32 2 2400 3600 1 16 1e-1 0 <br>
./build/test/mean_szp_2d ./data/ocean_uv.f32 2 2400 3600 1 16 1e-1 1 <br>
./build/test/mean_szp_2d ./data/ocean_uv.f32 2 2400 3600 1 16 1e-1 2 <br>

./build/test/mean_szp_1d ./data/hurricane_uv.f32 3 100 500 500 256 1e-1 0 <br>
./build/test/mean_szp_1d ./data/hurricane_uv.f32 3 100 500 500 256 1e-1 1 <br>
./build/test/mean_szp_1d ./data/hurricane_uv.f32 3 100 500 500 256 1e-1 2 <br>

./build/test/mean_szp_3d ./data/hurricane_uv.f32 3 100 500 500 16 1e-1 0 <br>
./build/test/mean_szp_3d ./data/hurricane_uv.f32 3 100 500 500 16 1e-1 1 <br>
./build/test/mean_szp_3d ./data/hurricane_uv.f32 3 100 500 500 16 1e-1 2 <br>

## Standard Deviation
./build/test/stddev_szp_1d ./data/ocean_uv.f32 2 2400 3600 1 256 1e-1 2 <br>
./build/test/stddev_szp_2d ./data/ocean_uv.f32 2 2400 3600 1 16 1e-1 2 <br>
./build/test/stddev_szp_1d ./data/hurricane_uv.f32 3 100 500 500 256 1e-1 2 <br>
./build/test/stddev_szp_3d ./data/hurricane_uv.f32 3 100 500 500 16 1e-1 2 <br>
