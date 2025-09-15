#pragma once
//#include <GL/gl.h> // Add this include at the top of the file
#define GLAD_GL_IMPLEMENTATION
#include <glad\glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW\glfw3.h>

#include <cuda_gl_interop.h>
#include "shader.h"

#define WIDTH 1280
#define HEIGHT 720

class VolumeRenderer{
public:
    VolumeRenderer(int gridSize, float density);
    ~VolumeRenderer();

    void initialize();
    void update();
    void render();

private:
    int gridSize;
    float density;
    unsigned int VBO, VAO, texture;
    Shader* shader;

    // CUDA resources
    cudaGraphicsResource* cudaTextureResource;
    unsigned long long* deviceData;
    size_t dataSize;

    void setupBuffers();
    void initializeCUDA();
    void cleanupCUDA();
};
