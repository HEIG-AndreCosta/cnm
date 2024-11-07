OBJS	= edge_detection_simd.o
SOURCE	= edge_detection_simd.cpp
HEADER	= 
OUT	= edge_detection_simd
CC	 = g++
FLAGS	 = -O3 -mcpu=cortex-a57 -ffast-math -c -Wall -I /usr/include/opencv4
LFLAGS	 = -lopencv_imgcodecs -lopencv_core
# -O3 enables almost all optimizations
# -mcpu specifies a target processor (and features)
# -ffast-math aggressive floating-point optimiuations
# -c flag generates object code for separate files
# -Wall enable all warnings

all: $(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(LFLAGS)


# create/compile the individual files >>separately<<
edge_detection_simd.o: edge_detection_simd.cpp
	$(CC) $(FLAGS) edge_detection_simd.cpp 


# clean house
clean:
	rm -f $(OBJS) $(OUT)