#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 faceData;  // x = energy or m_plus, y = m_minus

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

uniform mat3 inverseTransposeModel;

out vec4 normal_worldSpace;
out vec4 position_worldSpace;
out vec2 vFaceData;

void main() {
    normal_worldSpace   = vec4(normalize(inverseTransposeModel * normal), 0);
    position_worldSpace = vec4(position, 1.0);
    vFaceData = faceData;

    gl_Position = proj * view * model * vec4(position, 1.0);
}
