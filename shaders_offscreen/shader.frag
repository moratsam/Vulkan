#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMask;

void main() {
	outColor = texture(texSampler, fragTexCoord);
	outMask = vec4(0.0, 0.5, 0.0, 1.0);
}