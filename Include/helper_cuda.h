#pragma once

#include "cuda_runtime.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "curand.h"

// CUDA Error checking macros
#define checkCudaErrors(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define getLastCudaError(msg) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d after %s - %s\n", __FILE__, __LINE__, \
                    msg, cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static const char* __stdcall curandGetErrorString(curandStatus_t status){
    switch(status){
    case (CURAND_STATUS_SUCCESS): return "No errors";
    case (CURAND_STATUS_VERSION_MISMATCH): return "Header file and linked library version do not match";
    case (CURAND_STATUS_NOT_INITIALIZED): return "Generator not initialized";
    case (CURAND_STATUS_ALLOCATION_FAILED): return "Memory allocation failed";
    case (CURAND_STATUS_TYPE_ERROR): return "Generator is wrong type";
    case (CURAND_STATUS_OUT_OF_RANGE): return "Argument out of range";
    case (CURAND_STATUS_LENGTH_NOT_MULTIPLE): return "Length requested is not a multple of dimension";
    case (CURAND_STATUS_DOUBLE_PRECISION_REQUIRED): return "GPU does not have double precision required by MRG32k3a";
    case (CURAND_STATUS_LAUNCH_FAILURE): return "Kernel launch failure";
    case (CURAND_STATUS_PREEXISTING_FAILURE): return "Preexisting failure on library entry";
    case (CURAND_STATUS_INITIALIZATION_FAILED): return "Initialization of CUDA failed";
    case (CURAND_STATUS_ARCH_MISMATCH): return "Architecture mismatch, GPU does not support requested feature";
    case (CURAND_STATUS_INTERNAL_ERROR): return "Internal library error";
    default:        return "";
    }

}

#define checkCurandErrors(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "CURAND error at %s:%d - %s\n", __FILE__, __LINE__, \
                    curandGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device synchronization with error checking
inline void checkCudaDeviceSync() {
    checkCudaErrors(cudaDeviceSynchronize());
}

// Memory allocation helpers
template<typename T>
inline T* cudaMallocChecked(size_t count) {
    T* ptr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
inline void cudaFreeChecked(T* ptr) {
    if (ptr) {
        checkCudaErrors(cudaFree(ptr));
    }
}

// Memory copy helpers
template<typename T>
inline void cudaMemcpyH2D(T* d_dst, const T* h_src, size_t count) {
    checkCudaErrors(cudaMemcpy(d_dst, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
inline void cudaMemcpyD2H(T* h_dst, const T* d_src, size_t count) {
    checkCudaErrors(cudaMemcpy(h_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
inline void cudaMemcpyD2D(T* d_dst, const T* d_src, size_t count) {
    checkCudaErrors(cudaMemcpy(d_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// Memory set helper
template<typename T>
inline void cudaMemsetChecked(T* ptr, int value, size_t count) {
    checkCudaErrors(cudaMemset(ptr, value, count * sizeof(T)));
}

// CUDA device info
inline void printCudaDeviceInfo() {
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    
    printf("CUDA Device Count: %d\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
        
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global mem: %.1f MB\n", (float)deviceProp.totalGlobalMem / (1024*1024));
        printf("  Shared mem per block: %.1f KB\n", (float)deviceProp.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max thread dimensions: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Warp size: %d\n", deviceProp.warpSize);
    }
}

// Kernel launch configuration helpers
inline dim3 calculateGridSize(int totalThreads, int blockSize) {
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    return dim3(gridSize);
}

inline dim3 calculateGridSize2D(int2 totalThreads, dim3 blockSize) {
    int gridX = (totalThreads.x + blockSize.x - 1) / blockSize.x;
    int gridY = (totalThreads.y + blockSize.y - 1) / blockSize.y;
    return dim3(gridX, gridY);
}

// Timing utilities
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        checkCudaErrors(cudaEventCreate(&start_event));
        checkCudaErrors(cudaEventCreate(&stop_event));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        checkCudaErrors(cudaEventRecord(start_event, 0));
    }
    
    float stop() {
        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};