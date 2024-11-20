CC = gcc
CFLAGS = -fopenmp
LDFLAGS = -lm
TARGET = neural_network

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
