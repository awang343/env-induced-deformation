#version 330 core

in vec3 viewDir;
out vec4 fragColor;

uniform vec3 camPos;

void main() {
    vec3 color = vec3(0.75, 0.85, 1.0);
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
