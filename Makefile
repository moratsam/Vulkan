STB_INCLUDE_PATH = /usr/include/stb

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lXxf86vm -lXrandr -lXi
DUMP = dump#dump folder
OFFS_OUT = aut_offscreen #offscreen compilation out
VIEW_OUT = aut_view #view compilation out

Offscreen: offscreen.cpp
	g++ $(CFLAGS) -o $(OFFS_OUT) offscreen.cpp $(LDFLAGS)

View: view.cpp
	g++ $(CFLAGS) -o $(VIEW_OUT) view.cpp $(LDFLAGS)

.PHONY: test clean

offscreen: Offscreen
	if [ -d $(DUMP) ]; then				\
		if [ -f $(DUMP)/*ppm ]; then	\
			rm $(DUMP)/*ppm;				\
		fi;									\
	else										\
		mkdir $(DUMP);						\
	fi;												
	./$(OFFS_OUT)

view: View
	./$(VIEW_OUT)

clean:
	 rm $(OFFS_OUT) $(VIEW_OUT) $(DUMP)/*ppm; rmdir $(DUMP)
