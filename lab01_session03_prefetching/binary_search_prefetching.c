 #include <time.h>
 #include <stdio.h>
 #include <stdlib.h>

// Binary search function
// Parameters:
// - array: The sorted array to search within.
// - number_of_elements: The number of elements in the array.
// - key: The value to search for.
int binarySearch(int *array, int number_of_elements, int key) {
    int low = 0, high = number_of_elements - 1, mid;

    // Perform the binary search using a while loop
    while (low <= high) {
        mid = (low + high) / 2;  // Calculate the middle index

        // Optional prefetching hints if DO_PREFETCH is defined
        #ifdef DO_PREFETCH
        // Prefetch data in the low and high paths
        // This can help improve cache utilization and reduce memory latency
        __builtin_prefetch(&array[(mid + 1 + high) / 2], 0, 1);  // Low path prefetch
        __builtin_prefetch(&array[(low + mid - 1) / 2], 0, 1);   // High path prefetch
        #endif

        // Compare the element at the middle index with the key
        if (array[mid] < key)
            low = mid + 1;  // Adjust the search range to the upper half
        else if (array[mid] == key)
            return mid;  // Key found, return its index
        else if (array[mid] > key)
            high = mid - 1;  // Adjust the search range to the lower half
    }

    // Key not found in the array, return -1
    return -1;
}

int main() {
    // Define the size of the array
    int SIZE = 1024 * 1024 * 512;

    // Allocate memory for an array of integers
    int *array = malloc(SIZE * sizeof(int));

    // Initialize the array with values from 0 to SIZE-1
    for (int i = 0; i < SIZE; i++) {
        array[i] = i;
    }

    // Define the number of lookups to perform
    int NUM_LOOKUPS = 1024 * 1024 * 8;

    // Seed the random number generator
    srand(time(NULL));

    // Allocate memory for an array of random lookup values
    int *lookups = malloc(NUM_LOOKUPS * sizeof(int));

    // Generate random lookup values within the range of the array
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        lookups[i] = rand() % SIZE;
    }

    // Perform binary searches for the generated lookup values
    for (int i = 0; i < NUM_LOOKUPS; i++) {
        int result = binarySearch(array, SIZE, lookups[i]);
    }

    // Free the allocated memory for the array and lookup values
    free(array);
    free(lookups);
    
    return 0;
}