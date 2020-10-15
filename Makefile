STB_INCLUDE_PATH = /usr/include/stb
TINYOBJ_INCLUDE_PATH = /usr/include/tinyobjloader

CFLAGS = -std=c++17 -O3 -I$(STB_INCLUDE_PATH) -I$(TINYOBJ_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lXxf86vm -lXrandr -lXi

HelloTriangle: main.cpp
	g++ $(CFLAGS) -o triangle main.cpp $(LDFLAGS)

.PHONY: test clean

test: HelloTriangle
	./triangle

clean:
	shred -zun 6 triangle
