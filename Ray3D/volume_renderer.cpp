#include "volume_renderer.h"
#include "cuda_functions.h" // Добавьте эту строку
#include <cuda_gl_interop.h>
#include <iostream>
#include <helper_cuda.h>

extern void cuda_init(unsigned long long* data, int gridSize, float density);
extern void cuda_update(unsigned long long* data, int gridSize);
extern void cuda_render(cudaGraphicsResource* resource, unsigned long long* data,
                       int gridSize, int width, int height);

VolumeRenderer::VolumeRenderer(int gridSize, float density)
    : gridSize(gridSize), density(density), shader(nullptr),
    cudaTextureResource(nullptr), deviceData(nullptr){}

void VolumeRenderer::initialize(){
    shader = new Shader("shaders/vertex.vert", "shaders/fragment.frag");
    setupBuffers();
    initializeCUDA();
}

void VolumeRenderer::setupBuffers(){
    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

//void VolumeRenderer::initializeCUDA(){
//    dataSize = (gridSize * gridSize * gridSize + 63) / 64;
//    checkCudaErrors(cudaMalloc((void**)&deviceData, dataSize * sizeof(unsigned long long)));
//    cuda_init(deviceData, gridSize, density);
//
//    // Регистрируем как изображение
//    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture,
//                                                GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
//}
// ***
void VolumeRenderer::initializeCUDA(){
    dataSize = (gridSize * gridSize * gridSize + 63) / 64;
    checkCudaErrors(cudaMalloc((void**)&deviceData, dataSize * sizeof(unsigned long long)));
    
    // Проверка выделения памяти
    std::cout << "Allocated " << dataSize << " elements for device data density=" << density*100 << std::endl;
    
    cuda_init(deviceData, gridSize, density);
    
    // Проверка инициализации - посчитаем сколько точек создалось
    unsigned long long* hostData = new unsigned long long[dataSize];
    checkCudaErrors(cudaMemcpy(hostData, deviceData, dataSize * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    
    int pointCount = 0;
    for (int i = 0; i < dataSize; i++) {
        pointCount += __popcnt64(hostData[i]);
    }
    std::cout << "Initialized with " << pointCount << " points with " << (100.0 * pointCount)/ dataSize << "%" << std::endl;
    delete[] hostData;

        // Регистрируем как изображение
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture,
                                                GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}
// ***
void VolumeRenderer::update(){
    cuda_update(deviceData, gridSize);
}

void VolumeRenderer::render(){
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResource, 0));

    cuda_render(cudaTextureResource, deviceData, gridSize, WIDTH, HEIGHT);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResource, 0));

    shader->use();
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

VolumeRenderer::~VolumeRenderer(){
    cleanupCUDA();
    delete shader;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteTextures(1, &texture);
}

void VolumeRenderer::cleanupCUDA(){
    if(deviceData) 
        checkCudaErrors(cudaFree(deviceData));
    if(cudaTextureResource) 
        checkCudaErrors(cudaGraphicsUnregisterResource(cudaTextureResource));
}
