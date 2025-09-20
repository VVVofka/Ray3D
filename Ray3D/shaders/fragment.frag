#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main() {
    //FragColor = texture(screenTexture, TexCoord);
        // Простая проверка - выводим красный цвет если текстура работает
    vec4 texColor = texture(screenTexture, TexCoord);
    if (texColor.a == 0.0) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Красный если текстура пустая
    } else {
        FragColor = texColor;
    }

}