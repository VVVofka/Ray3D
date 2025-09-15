#pragma once
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

#ifdef __cplusplus
extern "C" {
#endif

    void cuda_init(unsigned long long* data, int gridSize, float density);
    void cuda_update(unsigned long long* data, int gridSize);
    void cuda_render(cudaGraphicsResource* resource, unsigned long long* data,
                     int gridSize, int width, int height);

#ifdef __cplusplus
}
#endif