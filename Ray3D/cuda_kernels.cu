#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>
#include "device_launch_parameters.h"
#include "cuda_functions.h"
#include <helper_cuda.h>

#define GRID_SIZE 1024
#define BLOCK_SIZE 256

__constant__ int c_gridSize = GRID_SIZE;
__device__ unsigned long long* d_data;

// Вспомогательные функции
__device__ float3 normalize(float3 v){
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if(length > 0.0f){
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }
    return v;
}

__device__ int3 wrap_position(int3 pos){
    pos.x = (pos.x + c_gridSize) % c_gridSize;
    pos.y = (pos.y + c_gridSize) % c_gridSize;
    pos.z = (pos.z + c_gridSize) % c_gridSize;
    return pos;
}

__device__ bool get_voxel(int3 pos){
    pos = wrap_position(pos);
    int linear_idx = pos.z * c_gridSize * c_gridSize + pos.y * c_gridSize + pos.x;
    int array_idx = linear_idx / 64;
    int bit_idx = linear_idx % 64;
    return (d_data[array_idx] >> bit_idx) & 1ULL;
}

// Атомарная OR операция для unsigned long long
__device__ unsigned long long atomicOrULL(unsigned long long* address, unsigned long long val){
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed | val);
    } while(assumed != old);

    return old;
}

// Альтернативная реализация записи в поверхность
__device__ void write_pixel(uchar4* output, int x, int y, int width, uchar4 color){
    if(x >= 0 && x < width && y >= 0){
        output[y * width + x] = color;
    }
}

__global__ void init_kernel(float density, curandState* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (c_gridSize * c_gridSize * c_gridSize + 63) / 64) return;

    curandState localState = states[idx];
    unsigned long long random_bits = 0;

    for(int i = 0; i < 64; i++){
        float rnd = curand_uniform(&localState);
        if(rnd < density){
            random_bits |= (1ULL << i);
        }
    }

    d_data[idx] = random_bits;
    states[idx] = localState;
}

__global__ void update_kernel(curandState* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (c_gridSize * c_gridSize * c_gridSize + 63) / 64) return;

    curandState localState = states[idx];
    unsigned long long current_bits = d_data[idx];

    // Сначала очищаем текущие биты
    d_data[idx] = 0;

    // Обрабатываем каждый установленный бит
    for(int bit = 0; bit < 64; bit++){
        if(current_bits & (1ULL << bit)){
            int linear_idx = idx * 64 + bit;
            int3 pos = {
                linear_idx % c_gridSize,
                (linear_idx / c_gridSize) % c_gridSize,
                linear_idx / (c_gridSize * c_gridSize)
            };

            // Генерируем случайное направление (0-5)
            int direction = curand(&localState) % 6;
            int3 new_pos = pos;

            switch(direction){
            case 0: new_pos.x = (new_pos.x + 1) % c_gridSize; break;
            case 1: new_pos.x = (new_pos.x - 1 + c_gridSize) % c_gridSize; break;
            case 2: new_pos.y = (new_pos.y + 1) % c_gridSize; break;
            case 3: new_pos.y = (new_pos.y - 1 + c_gridSize) % c_gridSize; break;
            case 4: new_pos.z = (new_pos.z + 1) % c_gridSize; break;
            case 5: new_pos.z = (new_pos.z - 1 + c_gridSize) % c_gridSize; break;
            }

            // Вычисляем новую позицию в массиве
            int new_linear_idx = new_pos.z * c_gridSize * c_gridSize +
                new_pos.y * c_gridSize + new_pos.x;
            int new_array_idx = new_linear_idx / 64;
            int new_bit_idx = new_linear_idx % 64;

            // Атомарно устанавливаем бит в новой позиции
            atomicOrULL(&d_data[new_array_idx], 1ULL << new_bit_idx);
        }
    }

    states[idx] = localState;
}

__global__ void render_kernel(cudaSurfaceObject_t surface, int width, int height, float time){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height) 
        return;

    // Вычисляем направление луча
    float aspect = (float)width / height;
    float u = (2.0f * x / width - 1.0f) * aspect;
    float v = 1.0f - 2.0f * y / height;

    // Камера с вращением
    float camAngle = time * 0.1f;
    float3 rayDir = normalize(make_float3(
        u * cosf(camAngle) - sinf(camAngle),
        v,
        -1.0f
    ));

    // Позиция камеры
    float3 cameraPos = make_float3(
        c_gridSize / 2.0f + sinf(time * 0.05f) * 100.0f,
        c_gridSize / 2.0f,
        c_gridSize * 2.0f
    );

    float3 rayPos = cameraPos;
    uchar4 color = make_uchar4(0, 0, 0, 255);

    // Ray marching
    for(int i = 0; i < 200; i++){
        int3 voxelPos = make_int3(
            (int)floorf(rayPos.x),
            (int)floorf(rayPos.y),
            (int)floorf(rayPos.z)
        );

        if(get_voxel(voxelPos)){
            // Жёлтый цвет
            color = make_uchar4(255, 255, 0, 255);
            break;
        }

        rayPos.x += rayDir.x * 1.5f;
        rayPos.y += rayDir.y * 1.5f;
        rayPos.z += rayDir.z * 1.5f;

        // Проверяем выход за границы
        if(rayPos.x < -100 || rayPos.x > c_gridSize + 100 ||
            rayPos.y < -100 || rayPos.y > c_gridSize + 100 ||
            rayPos.z < -100 || rayPos.z > c_gridSize + 100){
            break;
        }
    }

    // Записываем пиксель через surface
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

extern "C" void cuda_init(unsigned long long* data, int gridSize, float density){
    const void* p_gridSize = static_cast<const void*>(&c_gridSize); // Не удалять p_gridSize: используется для избежания красного подчёркивания в CUDACHECK
    checkCudaErrors(cudaMemcpyToSymbol(p_gridSize, &gridSize, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_data, &data, sizeof(unsigned long long*)));

    size_t num_elements = (gridSize * gridSize * gridSize + 63) / 64;
    size_t state_size = num_elements * sizeof(curandState);
    curandState* states;
    checkCudaErrors(cudaMalloc(&states, state_size));

    // Инициализируем генератор случайных чисел
    curandGenerator_t gen;
    checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, 1234));

    // Заполняем states случайными значениями
    float* temp;
    checkCudaErrors(cudaMalloc(&temp, state_size));
    checkCurandErrors(curandGenerateUniform(gen, temp, state_size / sizeof(float)));
    checkCudaErrors(cudaMemcpy(states, temp, state_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(temp));

    dim3 blocks(((unsigned)num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    init_kernel << <blocks, BLOCK_SIZE >> > (density, states);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(states));
    checkCurandErrors(curandDestroyGenerator(gen));
}

extern "C" void cuda_update(unsigned long long* data, int gridSize){
    checkCudaErrors(cudaMemcpyToSymbol(d_data, &data, sizeof(unsigned long long*)));

    size_t num_elements = (gridSize * gridSize * gridSize + 63) / 64;
    size_t state_size = num_elements * sizeof(curandState);
    curandState* states;
    checkCudaErrors(cudaMalloc(&states, state_size));

    curandGenerator_t gen;
    checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, clock()));

    float* temp;
    checkCudaErrors(cudaMalloc(&temp, state_size));
    checkCurandErrors(curandGenerateUniform(gen, temp, state_size / sizeof(float)));
    checkCudaErrors(cudaMemcpy(states, temp, state_size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(temp));

    dim3 blocks(((unsigned)num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    update_kernel << <blocks, BLOCK_SIZE >> > (states);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(states));
    checkCurandErrors(curandDestroyGenerator(gen));
}

extern "C" void cuda_render(cudaGraphicsResource* resource, unsigned long long* data,
                           int gridSize, int width, int height){
    checkCudaErrors(cudaMemcpyToSymbol(d_data, &data, sizeof(unsigned long long*)));

    // Получаем cudaArray для текстуры
    cudaArray* textureArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, resource, 0, 0));

    // Создаем surface object для записи
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObj;
    checkCudaErrors(cudaCreateSurfaceObject(&surfaceObj, &resDesc));

    // Рендерим сцену
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    static float time = 0.0f;
    time += 0.016f;

    // Используем surface для записи
    render_kernel << <gridDim, blockDim >> > (surfaceObj, width, height, time);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Удаляем surface object
    checkCudaErrors(cudaDestroySurfaceObject(surfaceObj));
}