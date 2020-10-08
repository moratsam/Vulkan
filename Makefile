CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lXxf86vm -lXrandr -lXi

HelloTriangle: main.cpp
	g++ $(CFLAGS) -o triangle main.cpp $(LDFLAGS)

.PHONY: test clean

test: HelloTriangle
	./triangle

clean:
	shred -zun 6 triangle
