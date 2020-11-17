STB_INCLUDE_PATH = /usr/include/stb

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lXxf86vm -lXrandr -lXi
DUMP = dump#dump folder
OFFS_OUT = aut_offscreen #offscreen compilation out
SWAP_OUT = aut_swap #swapchain compilation out

Offscreen: offscreen.cpp
	g++ $(CFLAGS) -o $(OFFS_OUT) offscreen.cpp $(LDFLAGS)

Swapchain: swapchain.cpp
	g++ $(CFLAGS) -o $(SWAP_OUT) swapchain.cpp $(LDFLAGS)

.PHONY: test clean

offscreen: Offscreen
	if [ -d $(DUMP) ]; then				\
		if [ -f $(DUMP)/offscreen*ppm ]; then	\
			rm $(DUMP)/offscreen*ppm;				\
		fi;									\
	else										\
		mkdir $(DUMP);						\
	fi;												
	./$(OFFS_OUT)

swapchain: Swapchain
	if [ -d $(DUMP) ]; then				\
		if [ -f $(DUMP)/screenshot*ppm ]; then	\
			rm $(DUMP)/screenshot*ppm;				\
		fi;									\
	else										\
		mkdir $(DUMP);						\
	fi;
	./$(SWAP_OUT)


clean:
	 rm $(OFFS_OUT) $(SWAP_OUT) $(DUMP)/*ppm; rmdir $(DUMP)
