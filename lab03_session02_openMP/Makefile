CC = gcc
CFLAGS = -fopenmp
LDFLAGS = -lm
TARGET = neural_network
NO_OPTI_TARGET = neural_network_no_opti

all: $(TARGET) $(NO_OPTI_TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(NO_OPTI_TARGET): main_noopti.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
