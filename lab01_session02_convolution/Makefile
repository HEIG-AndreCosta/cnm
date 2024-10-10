OBJS	= edge_detection.o
SOURCE	= edge_detection.cpp
HEADER	= 
OUT	= edge_detection
CC	 = g++
FLAGS	 =   -O0 -c -Wall -I /usr/include/opencv4
LFLAGS	 = -lopencv_imgcodecs -lopencv_core
# -O0 disables almost all optimizations
# -c flag generates object code for separate files
# -Wall enable all warnings

all: $(OBJS)
	$(CC) $(OBJS) -o $(OUT) $(LFLAGS)


# create/compile the individual files >>separately<<
edge_detection.o: edge_detection.cpp
	$(CC) $(FLAGS) edge_detection.cpp 


# clean house
clean:
	rm -f $(OBJS) $(OUT)