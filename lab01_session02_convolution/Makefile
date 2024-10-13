OBJS			= edge_detection_no_unroll.o
OBJS			+= edge_detection_manualo0_unroll.o
OBJS			+= edge_detection_manualo1_unroll.o
OBJS			+= edge_detection_manualo2_unroll.o

OBJS			+= edge_detection_compilero0_unroll.o
OBJS			+= edge_detection_compilero1_unroll.o
OBJS			+= edge_detection_compilero2_unroll.o
SOURCE			= edge_detection.cpp
HEADER			= 
OUT			= edge_detection
CC			= g++
NORMAL_FLAGS		= -O0 -c -Wall -I /usr/include/opencv4

MANUALO0_UNROLL_FLAGS	= -O0 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING
MANUALO1_UNROLL_FLAGS	= -O1 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING
MANUALO2_UNROLL_FLAGS	= -O2 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING

COMPILERO0_UNROLL_FLAGS	= -O0 -c -Wall -I /usr/include/opencv4 -funroll-loops
COMPILERO1_UNROLL_FLAGS	= -O1 -c -Wall -I /usr/include/opencv4 -funroll-loops
COMPILERO2_UNROLL_FLAGS	= -O2 -c -Wall -I /usr/include/opencv4 -funroll-loops

LFLAGS			= -lopencv_imgcodecs -lopencv_core


TARGETS = edge_detection_no_unroll

TARGETS += edge_detection_manualo0_unroll 
TARGETS += edge_detection_manualo1_unroll 
TARGETS += edge_detection_manualo2_unroll 

TARGETS += edge_detection_compilero0_unroll 
TARGETS += edge_detection_compilero1_unroll 
TARGETS += edge_detection_compilero2_unroll

# -O0 disables almost all optimizations
# -c flag generates object code for separate files
# -Wall enable all warnings

all: $(TARGETS)

edge_detection_no_unroll: edge_detection_no_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_manual_unroll: edge_detection_manual_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_manualo0_unroll: edge_detection_manualo0_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_manualo1_unroll: edge_detection_manualo1_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_manualo2_unroll: edge_detection_manualo2_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_compilero0_unroll: edge_detection_compilero0_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_compilero1_unroll: edge_detection_compilero1_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

edge_detection_compilero2_unroll: edge_detection_compilero2_unroll.o
	$(CC) $< -o $@ $(LFLAGS)

# create/compile the individual files >>separately<<
edge_detection_no_unroll.o: edge_detection.cpp
	$(CC) $(NORMAL_FLAGS) -o $@ $<


edge_detection_manualo0_unroll.o: edge_detection.cpp
	$(CC) $(MANUALO0_UNROLL_FLAGS) -o $@ $<

edge_detection_manualo1_unroll.o: edge_detection.cpp
	$(CC) $(MANUALO1_UNROLL_FLAGS) -o $@ $<

edge_detection_manualo2_unroll.o: edge_detection.cpp
	$(CC) $(MANUALO2_UNROLL_FLAGS) -o $@ $<

edge_detection_compilero0_unroll.o: edge_detection.cpp
	$(CC) $(COMPILERO0_UNROLL_FLAGS) -o $@ $<

edge_detection_compilero1_unroll.o: edge_detection.cpp
	$(CC) $(COMPILERO1_UNROLL_FLAGS) -o $@ $<

edge_detection_compilero2_unroll.o: edge_detection.cpp
	$(CC) $(COMPILERO2_UNROLL_FLAGS) -o $@ $<

#clean house
clean:
	rm -f $(OBJS) $(TARGETS)
