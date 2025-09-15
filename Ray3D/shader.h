#pragma once
#include <string>

class Shader{
public:
    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();

    void use();
    unsigned int getID() const{ return ID; }

private:
    unsigned int ID;

    void checkCompileErrors(unsigned int shader, std::string type);
};
