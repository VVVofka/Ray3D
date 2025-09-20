#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main() {
    //FragColor = texture(screenTexture, TexCoord);
        // ������� �������� - ������� ������� ���� ���� �������� ��������
    vec4 texColor = texture(screenTexture, TexCoord);
    if (texColor.a == 0.0) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0); // ������� ���� �������� ������
    } else {
        FragColor = texColor;
    }

}