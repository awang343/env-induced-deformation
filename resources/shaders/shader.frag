#version 330 core
out vec4 fragColor;

in vec4 normal_worldSpace;
in vec4 position_worldSpace;

uniform int wire = 0;
uniform float red = 1.0;
uniform float green = 1.0;
uniform float blue = 1.0;
uniform float alpha = 1.0;

void main() {
    if (wire == 1) {
        fragColor = vec4(0.1, 0.1, 0.1, 1.0);
        return;
    }

    vec3 N = normalize(normal_worldSpace.xyz);
    // Thin shell: flip the normal on back-facing fragments so both
    // sides of the surface get lit consistently.
    if (!gl_FrontFacing) N = -N;
    vec3 baseColor = vec3(red, green, blue);

    // Two-point lighting
    vec3 lightPos1 = vec3(3.0, 5.0, -4.0);
    vec3 lightColor1 = vec3(1.0, 0.85, 0.65);

    vec3 lightPos2 = vec3(-3.0, 2.0, 3.0);
    vec3 lightColor2 = vec3(0.5, 0.4, 0.35);

    vec3 pos = position_worldSpace.xyz;

    // Diffuse
    vec3 L1 = normalize(lightPos1 - pos);
    vec3 L2 = normalize(lightPos2 - pos);
    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);

    // Low ambient so shadows have contrast
    vec3 ambient = vec3(0.08, 0.07, 0.06);

    vec3 color = ambient * baseColor
               + diff1 * lightColor1 * baseColor * 0.8
               + diff2 * lightColor2 * baseColor * 0.4;

    // Gamma correction
    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));

    fragColor = vec4(color, alpha);
}
