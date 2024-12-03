#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define INPUT_SIZE	 784
#define HIDDEN_SIZE	 128
#define OUTPUT_SIZE	 10

#define TEST_INPUT_SIZE	 4
#define TEST_HIDDEN_SIZE 3
#define TEST_OUTPUT_SIZE 2

#define LEARNING_RATE	 0.01
#define NUM_EPOCHS	 1
#define NUM_SAMPLES	 60

// Neural network structure
typedef struct {
	double input[INPUT_SIZE];
	double hidden[HIDDEN_SIZE];
	double output[OUTPUT_SIZE];
	double weights_ih[INPUT_SIZE][HIDDEN_SIZE];
	double weights_ho[HIDDEN_SIZE][OUTPUT_SIZE];
} NeuralNetwork;

// Function declarations
void initialize_network(NeuralNetwork *network);
void forward_pass(NeuralNetwork *network);
void backpropagation(NeuralNetwork *network, double target[OUTPUT_SIZE]);

void test_network()
{
	NeuralNetwork network;

	// Fixed input and weights
	double test_input[TEST_INPUT_SIZE] = { 1.0, 0.5, -0.5, 0.0 };
	double test_target[TEST_OUTPUT_SIZE] = { 0.0, 1.0 };

	double test_weights_ih[TEST_INPUT_SIZE][TEST_HIDDEN_SIZE] = {
		{ 0.1, -0.2, 0.3 },
		{ 0.4, 0.5, -0.6 },
		{ -0.7, 0.8, -0.9 },
		{ 0.1, -0.4, 0.6 }
	};

	double test_weights_ho[TEST_HIDDEN_SIZE][TEST_OUTPUT_SIZE] = {
		{ 0.2, -0.3 }, { 0.4, 0.5 }, { -0.5, 0.7 }
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

void generate_dummy_data(double training_data[NUM_SAMPLES][INPUT_SIZE],
			 double training_labels[NUM_SAMPLES][OUTPUT_SIZE])
{
	for (int i = 0; i < NUM_SAMPLES; ++i) {
		// Dummy input data
		for (int j = 0; j < INPUT_SIZE; ++j) {
			training_data[i][j] =
				((double)rand() / RAND_MAX) * 2.0 -
				1.0; // Random values between -1 and 1
		}
		// Dummy target output (replace with your actual labels)
		int true_class =
			rand() %
			OUTPUT_SIZE; // Random class between 0 and OUTPUT_SIZE - 1
		for (int k = 0; k < OUTPUT_SIZE; ++k) {
			training_labels[i][k] = (k == true_class) ? 1.0 : 0.0;
		}
	}
}

int main(int argc, char **argv)
{
	if (argc == 2) {
		test_network();
		return 0;
	}
	NeuralNetwork network;
	initialize_network(&network);

	double training_data[NUM_SAMPLES][INPUT_SIZE];
	double training_labels[NUM_SAMPLES][OUTPUT_SIZE];
	generate_dummy_data(training_data, training_labels);

	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);
	// Training loop
	for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
		for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
			// Load input data and target output for the current sample
			double input_data[INPUT_SIZE];
			double target[OUTPUT_SIZE];

#pragma omp parallel
			{
#pragma omp for nowait
				for (int i = 0; i < INPUT_SIZE; ++i) {
					input_data[i] =
						training_data[sample][i];
				}
#pragma omp for nowait
				for (int i = 0; i < OUTPUT_SIZE; ++i) {
					target[i] = training_labels[sample][i];
				}
			}
			// Forward pass
			forward_pass(&network);

			// Backpropagation
			backpropagation(&network, target);
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("Prediction time: %fs\n", elapsed);
	return 0;
}

/**
 * Initializes the neural network with small random values for weights.
 *
 * @param network Pointer to the NeuralNetwork structure to be initialized.
 */
void initialize_network(NeuralNetwork *network)
{
	for (int i = 0; i < INPUT_SIZE; ++i) {
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			network->weights_ih[i][j] =
				(((double)rand()) / RAND_MAX) *
				sqrt(2. / (INPUT_SIZE + OUTPUT_SIZE));
		}
	}

	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < OUTPUT_SIZE; ++j) {
			network->weights_ho[i][j] =
				(((double)rand()) / RAND_MAX) *
				sqrt(2. / (INPUT_SIZE + OUTPUT_SIZE));
		}
	}
}
double sigmoid(double input)
{
	return 1. / (1. + exp(-input));
}

/**
 * Performs the forward pass of the neural network.
 *
 * @param network Pointer to the NeuralNetwork structure.
 */
void forward_pass(NeuralNetwork *network)
{
#pragma omp parallel for
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		int sum = 0;
#pragma omp parallel for reduction(+ : sum)
		for (int j = 0; j < INPUT_SIZE; ++j) {
			sum += network->input[j] * network->weights_ih[j][i];
		}
		network->hidden[i] = sigmoid(sum);
	}

#pragma omp parallel for
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		int sum = 0;
#pragma omp parallel for reduction(+ : sum)
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			sum += network->hidden[j] * network->weights_ho[j][i];
		}
		network->output[i] = sigmoid(sum);
	}
}

/**
 * Performs backpropagation and updates the weights of the neural network.
 *
 * @param network Pointer to the NeuralNetwork structure.
 * @param target Array representing the target output for the current sample.
 */
void backpropagation(NeuralNetwork *network, double target[OUTPUT_SIZE])
{
	double *output_gradients =
		calloc(OUTPUT_SIZE, sizeof(output_gradients));
	assert(output_gradients);
#pragma omp parallel for
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		output_gradients[i] = (target[i] - network->output[i]) *
				      network->output[i] *
				      (1 - network->output[i]);
#pragma omp parallel for
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			network->weights_ho[j][i] += LEARNING_RATE *
						     output_gradients[i] *
						     network->hidden[j];
		}
	}
#pragma omp parallel for
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		int sum = 0;
		network->hidden[i] = 0;
#pragma omp parallel for reduction(+ : sum)
		for (int j = 0; j < OUTPUT_SIZE; ++j) {
			sum += output_gradients[j] * network->weights_ho[i][j] *
			       network->hidden[i] * (1 - network->hidden[i]);
		}
		network->hidden[i] = sum;
	}

#pragma omp parallel for
	for (int i = 0; i < INPUT_SIZE; ++i) {
#pragma omp parallel for
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			network->weights_ih[i][j] += LEARNING_RATE *
						     network->hidden[j] *
						     network->input[i];
		}
	}
}
