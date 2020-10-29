#!/bin/bash
glslc shader.frag -o frag.spv
glslc shader.vert -o vert.spv
glslc mask_shader.frag -o mask_frag.spv
glslc mask_shader.vert -o mask_vert.spv
