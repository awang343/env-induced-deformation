#version 330 core
out vec4 fragColor;

in vec4 normal_worldSpace;
in vec4 position_worldSpace;
in float vEnergy;

uniform int displayMode = 0;  // 0=solid, 1=heatmap, 2=wireframe
uniform float red = 1.0;
uniform float green = 1.0;
uniform float blue = 1.0;
uniform float alpha = 1.0;

// Turbo-ish colormap: blue → cyan → green → yellow → red
vec3 heatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (t < 0.25) {
        c = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 1.0), t / 0.25);
    } else if (t < 0.5) {
        c = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.2), (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        c = mix(vec3(0.0, 1.0, 0.2), vec3(1.0, 1.0, 0.0), (t - 0.5) / 0.25);
    } else {
        c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) / 0.25);
    }
    return c;
}

void main() {
    if (displayMode == 2) {
        fragColor = vec4(0.1, 0.1, 0.1, 1.0);
        return;
    }

    vec3 N = normalize(normal_worldSpace.xyz);
    if (!gl_FrontFacing) N = -N;

    // Two-point lighting
    vec3 lightPos1 = vec3(3.0, 5.0, -4.0);
    vec3 lightColor1 = vec3(1.0, 0.85, 0.65);
    vec3 lightPos2 = vec3(-3.0, 2.0, 3.0);
    vec3 lightColor2 = vec3(0.5, 0.4, 0.35);
    vec3 pos = position_worldSpace.xyz;
    vec3 L1 = normalize(lightPos1 - pos);
    vec3 L2 = normalize(lightPos2 - pos);
    float diff1 = max(dot(N, L1), 0.0);
    float diff2 = max(dot(N, L2), 0.0);
    vec3 ambient = vec3(0.08, 0.07, 0.06);

    vec3 baseColor;
    if (displayMode == 1) {
        // Energy heatmap. vEnergy is pre-normalized: 0 = zero, 1 = initial max.
        float e = clamp(vEnergy, 0.0, 1.0);
        baseColor = heatmap(e);
    } else {
        baseColor = vec3(red, green, blue);
    }

    vec3 color = ambient * baseColor
               + diff1 * lightColor1 * baseColor * 0.8
               + diff2 * lightColor2 * baseColor * 0.4;

    color = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
    fragColor = vec4(color, alpha);
}
