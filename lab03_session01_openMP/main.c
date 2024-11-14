#include <stdio.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include "wdbc.h"

#define MAX_CHAR 250

#define DATA_SIZE  569
#define TRAIN_SIZE 400
#define TEST_SIZE  DATA_SIZE - TRAIN_SIZE

#define NB_THREADS 4

/*
    Calculates the distance between an array of elements and one single element

    @param data Array of elements
    @param a Pointer to single element
    @param data_size Size of data
    @param Array of distances
*/
void calc_distance(const WBCD* data, const WBCD* a, const int data_size, double* distances)
{
        
    for (int i = 0; i < data_size; i++) {
        distances[i] = ecludian_distance(&data[i], a);
    }

}

/*
    Calculates the index of the k minimum values

    @param distances Array of values
    @param dist_size Size of the array of values
    @param kmin Array of indeces
    @param k_size Size of the array of indexes
*/
void kargmin(const double* distances, int dist_size, int* kmin, int k_size)
{
    double pre_min_disntace = 0;

    for(int k_idx = 0; k_idx < k_size; ++k_idx)
    {
        double cur_min_distance = DBL_MAX;          

        for(int dist_idx = 0; dist_idx < dist_size; ++dist_idx)
        {        
            if(distances[dist_idx] < cur_min_distance && distances[dist_idx] > pre_min_disntace)
            {
                kmin[k_idx] = dist_idx;
                cur_min_distance = distances[dist_idx];                
            }
        }

        pre_min_disntace = cur_min_distance;
    }
}

/*
    Predict label using kNN

    @param train Array of train data
    @param train_size Number of elements in array train
    @param test Array of test data
    @param test_size Number of elements in test size
    @param k Number of nearest neighborgs
    @param predictions Array of label prediction
*/
void predict(const WBCD* train, int train_size, const WBCD* test, int test_size, int k, char* predictions){

    double* distances = (double*)calloc(train_size, sizeof(double));
    int* kNN = (int*)calloc(k, sizeof(int));

    int label_b_count, label_m_count;  
    
    for (int test_idx = 0; test_idx < test_size; ++test_idx)
    {
        // Calculate distances
        calc_distance(train, &test[test_idx], train_size, distances);
        // Calculate k closest elements
        kargmin(distances, train_size, kNN, k);

        // Count kNN labels
        label_b_count = 0;
        label_m_count = 0;

        for(int k_idx = 0; k_idx < k; ++k_idx)
        {
            if(train[kNN[k_idx]].diagnosis == 'B')
            {
                label_b_count = label_b_count + 1;
            }else{
                label_m_count = label_m_count + 1;
            }
        }

        // Voting
        if(label_b_count > label_m_count){
            predictions[test_idx] = 'B';
        }else{
            predictions[test_idx] = 'M';
        }
    }

    free(distances);
    free(kNN);
}


void read_data(const char *filename, WBCD* data)
{
    FILE *fp;
    char row[MAX_CHAR];

    fp = fopen(filename,"r");

    for( int line_idx = 0; line_idx < DATA_SIZE; ++line_idx)
    {
        fgets(row, MAX_CHAR, fp);        
        parse_wbcd_line(row, &data[line_idx]);               
    }
}

int main( int argc, char **argv )
{   
    int k;
    char *end_opt_parser, *filename;

    if(argc == 3)
    {
        k = (int)(strtoul(argv[2], &end_opt_parser, 10));
        if(*end_opt_parser != '\0')
        {
            fprintf(stderr, "Error parsing k '%s'", argv[2]);
            return EXIT_FAILURE;
        } 
        filename = argv[1];
    } else {
        fprintf(stdout, "Usage %s FILENAME K \n", argv[0]);
        return EXIT_FAILURE;
    }

    // Read data from file
    WBCD data[DATA_SIZE];
    read_data(filename, data);

    // Split data into train and test
    WBCD *train = &data[0];
    WBCD *test  = &data[TRAIN_SIZE];

    // Predict test labels using kNN
    char label_prediction[TEST_SIZE];   

    // https://stackoverflow.com/questions/2962785/c-using-clock-to-measure-time-in-multi-threaded-programs
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);
    predict(train, TRAIN_SIZE, test, TEST_SIZE, k, label_prediction);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Prediction time: %fs\n", elapsed);

    // Calculate confusion matrix
    int confusion_matrix[4] = {
        0,  // True benign (correct classified as bening)
        0,  // False malignant (classified as malignant but it is bening)
        0,  // False bening (classifed as bening but it is malignat)
        0   // True malignant (correct classified as malignant)
    };
    
    for (int test_idx = 0; test_idx < TEST_SIZE; ++test_idx)
    {        
        char label_real = data[TRAIN_SIZE+test_idx].diagnosis;
        char label_pred = label_prediction[test_idx];

        if(label_real == 'B' && label_pred == 'B'){
            confusion_matrix[0] = confusion_matrix[0] + 1;
        }

        if(label_real == 'B' && label_pred == 'M'){
            confusion_matrix[1] = confusion_matrix[1] + 1;
        }

        if(label_real == 'M' && label_pred == 'B'){
            confusion_matrix[2] = confusion_matrix[2] + 1;
        }

        if(label_real == 'M' && label_pred == 'M'){
            confusion_matrix[3] = confusion_matrix[3] + 1;
        }

    }

    printf("|   |  Be |  Ma |\n");
    printf("|---|-----|-----|\n");
    printf("|Be | %3d | %3d |\n", confusion_matrix[0], confusion_matrix[1]);
    printf("|Ma | %3d | %3d |\n", confusion_matrix[2], confusion_matrix[3]);

    return EXIT_SUCCESS;
}
