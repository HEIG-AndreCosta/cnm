OBJS	= main.o
SOURCE	= main.c
HEADER	= wdbc.h
OUT	= knn
CC	 = gcc
FLAGS	 = -g -c -Wall -fopenmp
LFLAGS	 = -lm 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.c
	$(CC) $(FLAGS) main.c 


clean:
	rm -f $(OBJS) $(OUT)
