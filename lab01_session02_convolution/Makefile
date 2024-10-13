OBJS			= edge_detection_no_unroll.o
OBJS			+= edge_detection_manual_unroll.o
OBJS			+= edge_detection_compiler_unroll.o
SOURCE			= edge_detection.cpp
HEADER			= 
OUT			= edge_detection
CC			= g++
NORMAL_FLAGS		= -O0 -c -Wall -I /usr/include/opencv4
MANUAL_UNROLL_FLAGS	= -O0 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING
COMPILER_UNROLL_FLAGS	= -O0 -c -Wall -I /usr/include/opencv4 -funroll-loops
LFLAGS			= -lopencv_imgcodecs -lopencv_core


# -O0 disables almost all optimizations
# -c flag generates object code for separate files
# -Wall enable all warnings

all: edge_detection_no_unroll edge_detection_manual_unroll edge_detection_compiler_unroll

edge_detection_no_unroll: edge_detection_no_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_manual_unroll: edge_detection_manual_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_compiler_unroll: edge_detection_compiler_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

# create/compile the individual files >>separately<<
edge_detection_no_unroll.o: edge_detection.cpp
	$(CC) $(NORMAL_FLAGS) -o $@ $<

edge_detection_manual_unroll.o: edge_detection.cpp
	$(CC) $(MANUAL_UNROLL_FLAGS) -o $@ $<

edge_detection_compiler_unroll.o: edge_detection.cpp
	$(CC) $(COMPILER_UNROLL_FLAGS) -o $@ $<

# clean house
clean:
	rm -f $(OBJS) $(OUT)
