#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#define TEST_INPUT_SIZE 4
#define TEST_HIDDEN_SIZE 3
#define TEST_OUTPUT_SIZE 2
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define NUM_EPOCHS 1
#define NUM_SAMPLES 60

// Neural network structure
typedef struct {
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
    double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE];
} NeuralNetwork;

// Function declarations
void initialize_network(NeuralNetwork* network);
void forward_pass(NeuralNetwork* network);
void backpropagation(NeuralNetwork* network, double target[OUTPUT_SIZE]);

void test_network() {
    NeuralNetwork network;

    // Fixed input and weights
    double test_input[TEST_INPUT_SIZE] = {1.0, 0.5, -0.5, 0.0};
    double test_target[TEST_OUTPUT_SIZE] = {0.0, 1.0};

    double test_weights_ih[TEST_INPUT_SIZE][TEST_HIDDEN_SIZE] = {
        {0.1, -0.2, 0.3},
        {0.4, 0.5, -0.6},
        {-0.7, 0.8, -0.9},
        {0.1, -0.4, 0.6}
    };

    double test_weights_ho[TEST_HIDDEN_SIZE][TEST_OUTPUT_SIZE] = {
        {0.2, -0.3},
        {0.4, 0.5},
        {-0.5, 0.7}
    };

    // Initialize the network
    for (int i = 0; i < TEST_INPUT_SIZE; ++i) {
        network.input[i] = test_input[i];
    }

    for (int i = 0; i < TEST_INPUT_SIZE; ++i) {
        for (int j = 0; j < TEST_HIDDEN_SIZE; ++j) {
            network.weights_ih[i][j] = test_weights_ih[i][j];
        }
    }

    for (int j = 0; j < TEST_HIDDEN_SIZE; ++j) {
        for (int k = 0; k < TEST_OUTPUT_SIZE; ++k) {
            network.weights_ho[j][k] = test_weights_ho[j][k];
        }
    }

    // Perform forward pass
    forward_pass(&network);

    // Print results of forward pass
    printf("Forward pass output:\n");
    for (int k = 0; k < TEST_OUTPUT_SIZE; ++k) {
        printf("Output[%d]: %.6f\n", k, network.output[k]);
    }

    // Perform backpropagation
    backpropagation(&network, test_target);

    // Print updated weights
    printf("\nUpdated weights (input to hidden):\n");
    for (int i = 0; i < TEST_INPUT_SIZE; ++i) {
        for (int j = 0; j < TEST_HIDDEN_SIZE; ++j) {
            printf("%.6f ", network.weights_ih[i][j]);
        }
        printf("\n");
    }

    printf("\nUpdated weights (hidden to output):\n");
    for (int j = 0; j < TEST_HIDDEN_SIZE; ++j) {
        for (int k = 0; k < TEST_OUTPUT_SIZE; ++k) {
            printf("%.6f ", network.weights_ho[j][k]);
        }
        printf("\n");
    }
}

void generate_dummy_data(double training_data[NUM_SAMPLES][INPUT_SIZE], double training_labels[NUM_SAMPLES][OUTPUT_SIZE]) {
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        // Dummy input data
        for (int j = 0; j < INPUT_SIZE; ++j) {
            training_data[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
        }

        // Dummy target output (replace with your actual labels)
        int true_class = rand() % OUTPUT_SIZE; // Random class between 0 and OUTPUT_SIZE - 1
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            training_labels[i][k] = (k == true_class) ? 1.0 : 0.0;
        }
    }
}

int main() {

    NeuralNetwork network;
    initialize_network(&network);

    double training_data[NUM_SAMPLES][INPUT_SIZE];
    double training_labels[NUM_SAMPLES][OUTPUT_SIZE];
    generate_dummy_data(training_data, training_labels);

    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {

        for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
            // Load input data and target output for the current sample
            double input_data[INPUT_SIZE];
            double target[OUTPUT_SIZE];

            for (int i = 0; i < INPUT_SIZE; ++i) {
                network.input[i] = training_data[sample][i];
            }
            
            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                target[i] = training_labels[sample][i];
            }
            // Forward pass
            forward_pass(&network);

            // Backpropagation
            backpropagation(&network, target); 
    }

    // Print final results

    return 0;
    }
}

/**
 * Initializes the neural network with small random values for weights.
 *
 * @param network Pointer to the NeuralNetwork structure to be initialized.
 */
void initialize_network(NeuralNetwork* network) {
    // TODO: Implement the initialization of weights with small random values.
    // You need to set random values for network->weights_ih and network->weights_ho.
}

/**
 * Performs the forward pass of the neural network.
 *
 * @param network Pointer to the NeuralNetwork structure.
 */
void forward_pass(NeuralNetwork* network) {
    // TODO: Implement the forward pass, including applying the sigmoid activation function.
    // The results should be stored in network->hidden and network->output.
}

/**
 * Performs backpropagation and updates the weights of the neural network.
 *
 * @param network Pointer to the NeuralNetwork structure.
 * @param target Array representing the target output for the current sample.
 */
void backpropagation(NeuralNetwork* network, double target[OUTPUT_SIZE]) {
    // TODO: Implement backpropagation to compute gradients and update weights.
    // The results should be stored in network->weights_ih and network->weights_ho.
}

