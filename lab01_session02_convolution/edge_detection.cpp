#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>

using namespace cv;

int sobel_x[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };

int sobel_y[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

int main(int argc, char const *argv[])
{
	if (argc < 2) {
		std::cout << "Usage " << argv[0] << " image_file" << std::endl;
		return EXIT_FAILURE;
	}
	std::string image_file = std::string(argv[1]);
	// Get filename without extention
	std::string base_name =
		argc != 3 ? image_file.substr(0, image_file.find_last_of('.')) :
			    argv[2];
	// Open image as grayscale as OpenCV Mat
	Mat src = imread(image_file, IMREAD_GRAYSCALE);

	if (src.empty()) {
		std::cout << "Error reading image \"" << argv[1] << "\""
			  << std::endl;
		return EXIT_FAILURE;
	}

	if (src.type() != CV_8UC1) {
		std::cout << "Error reading image as grayscale " << argv[1]
			  << std::endl;
		return EXIT_FAILURE;
	}

	int image_rows = src.rows;
	int image_cols = src.cols;

	// Cast image data to pointer of unsigned chars because image is read as grayscale (CV_8UC1)
	uchar *src_data = reinterpret_cast<uchar *>(src.data);

	// Copy image data to matrix
	uchar A[image_rows][image_cols];
	std::memcpy(A, src_data, image_rows * image_cols * sizeof(uchar));

	// Horizontal gradient
	int Gx[image_rows][image_cols];
	// Vertical gradient
	int Gy[image_rows][image_cols];

	// Start measuring time
	auto start = std::chrono::high_resolution_clock::now();

/* **************************************** CONVOLUTION ************************************** */
/*
        Iterate over image
        Notice i and j values. We ignore the edges to avoid some complexity.
        YOUR LOOP UNROLLING
    */
#ifdef LOOP_UNROLLING

	for (int i = 1; i < image_rows - 1; ++i) {
		for (int j = 1; j < image_cols - 1; ++j) {
			Gx[i][j] += sobel_x[0][0] * A[0][0];
			Gy[i][j] += sobel_y[0][0] * A[0][0];
			Gx[i][j] += sobel_x[0][1] * A[0][1];
			Gy[i][j] += sobel_y[0][1] * A[0][1];
			Gx[i][j] += sobel_x[0][2] * A[0][2];
			Gy[i][j] += sobel_y[0][2] * A[0][2];

			Gx[i][j] += sobel_x[1][0] * A[1][0];
			Gy[i][j] += sobel_y[1][0] * A[1][0];
			Gx[i][j] += sobel_x[1][1] * A[1][1];
			Gy[i][j] += sobel_y[1][1] * A[1][1];
			Gx[i][j] += sobel_x[1][2] * A[1][2];
			Gy[i][j] += sobel_y[1][2] * A[1][2];
			Gx[i][j] += sobel_x[2][0] * A[2][0];
			Gy[i][j] += sobel_y[2][0] * A[2][0];
			Gx[i][j] += sobel_x[2][1] * A[2][1];
			Gy[i][j] += sobel_y[2][1] * A[2][1];
			Gx[i][j] += sobel_x[2][2] * A[2][2];
			Gy[i][j] += sobel_y[2][2] * A[2][2];
		}
	}

#else
	for (int i = 1; i < image_rows - 1; ++i) {
		for (int j = 1; j < image_cols - 1; ++j) {
			//Iterate over kernel
			for (int ki = 0; ki < 3; ++ki) {
				for (int kj = 0; kj < 3; ++kj) {
					Gx[i][j] +=
						sobel_x[ki][kj] *
						A[i + (ki - 1)][j + (kj - 1)];
					Gy[i][j] +=
						sobel_y[ki][kj] *
						A[i + (ki - 1)][j + (kj - 1)];
				}
			}
		}
	}
#endif

	/* **************************************************************************************** */

	//Stop measuring time
	auto end = std::chrono::high_resolution_clock::now();

	// Total gradient
	uchar G[image_rows][image_cols];

	for (int i = 0; i < image_rows; ++i) {
		for (int j = 0; j < image_cols; ++j) {
			// Gradient is the square root of the sum of the squared gradients
			double value = std::sqrt(std::pow(Gx[i][j], 2) +
						 std::pow(Gy[i][j], 2));
			// Trunk value into range
			G[i][j] = std::max(0., std::min(255., value));
		}
	}

	// Calculate how long took convolution
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
		end - start);

	std::cout << image_rows << ", " << image_cols << ", "
		  << duration.count() << std::endl;

	// Save gradients as images
	imwrite(base_name + "_x.png",
		Mat(image_rows, image_cols, CV_32SC1, Gx));
	imwrite(base_name + "_y.png",
		Mat(image_rows, image_cols, CV_32SC1, Gy));
	imwrite(base_name + "_edges.png",
		Mat(image_rows, image_cols, CV_8UC1, G));

	return EXIT_SUCCESS;
}
