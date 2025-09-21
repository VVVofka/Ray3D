//#include <GLFW/glfw3.h>
//#include <cuda_gl_interop.h>
#include "shader.h"
#include "volume_renderer.h"
VolumeRenderer* volumeRenderer = nullptr;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

int main(){
    if(!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Volume Rendering", NULL, NULL);
    if(!window){
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        return -1;
    }

    volumeRenderer = new VolumeRenderer(1024, 0.0001f);
    volumeRenderer->initialize();

    while(!glfwWindowShouldClose(window)){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        volumeRenderer->update();
        volumeRenderer->render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    delete volumeRenderer;
    glfwTerminate();
    return 0;
}
