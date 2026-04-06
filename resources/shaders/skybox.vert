#version 330 core

layout(location = 0) in vec2 position;

uniform mat4 invProjView;

out vec3 viewDir;

void main() {
    gl_Position = vec4(position, 1.0, 1.0);
    // Unproject from clip space to world direction
    vec4 worldPos = invProjView * vec4(position, 1.0, 1.0);
    viewDir = worldPos.xyz / worldPos.w;
}
