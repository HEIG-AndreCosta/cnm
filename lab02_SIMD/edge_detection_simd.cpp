#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>

#include <arm_neon.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int16x8_t sobel_x = { 1, 0, -1, 2, -2, 1, 0, -1 };

int16x8_t sobel_y = { 1, 2, 1, 0, 0, -1, -2, -1 };

int main(int argc, char const *argv[])
{
	if (argc != 2) {
		std::cout << "Usage " << argv[0] << " image_file" << std::endl;
		return EXIT_FAILURE;
	}
	std::string image_file = std::string(argv[1]);
	// Get filename without extention
	std::string base_name =
		image_file.substr(0, image_file.find_last_of('.'));
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

	size_t image_rows = src.rows;
	size_t image_cols = src.cols;

	//Convert image to from uint8_t to int16_t
	Mat img_16s;
	src.convertTo(img_16s, CV_16S);

	int16_t *src_data = reinterpret_cast<int16_t *>(img_16s.data);

	int16_t A[image_rows][image_cols];

	std::memcpy(A, src_data, image_rows * image_cols * sizeof(int16_t));

	// Horizontal gradient
	int16_t Gx[image_rows][image_cols];
	// Vertical gradient
	int16_t Gy[image_rows][image_cols];

	/* **************************************** CONVOLUTION ************************************** */

	int16x8_t gradient_x, gradient_y;

	for (int i = 1; i < image_rows - 1; ++i) {
		for (int j = 1; j < image_cols - 1; ++j) {
			int16x8_t gradient_y = {
				A[i - 1][j - 1], A[i - 1][j],
				A[i - 1][j + 1], A[i][j - 1],
				A[i][j + 1],	 A[i + 1][j - 1],
				A[i + 1][j],	 A[i + 1][j + 1]

			};

			int16x8_t gradient_x = {
				A[i - 1][j - 1], A[i - 1][j],
				A[i - 1][j + 1], A[i][j - 1],
				A[i][j + 1],	 A[i + 1][j - 1],
				A[i + 1][j],	 A[i + 1][j + 1]

			};
			gradient_x = vmulq_s16(gradient_x, sobel_x);
			gradient_y = vmulq_s16(gradient_y, sobel_y);
			Gx[i][j] = vaddvq_s16(gradient_x);
			Gy[i][j] = vaddvq_s16(gradient_y);
		}
	}
	// Implement the convolution algorithm using NEON intrinsics

	/* **************************************************************************************** */

	// Total gradient
	uint8_t G[image_rows][image_cols];

	for (size_t i = 0; i < image_rows; ++i) {
		for (size_t j = 0; j < image_cols; ++j) {
			// Gradient is the square root of the sum of the squared gradients
			double value = std::sqrt(std::pow(Gx[i][j], 2) +
						 std::pow(Gy[i][j], 2));
			// Trunk value into range
			G[i][j] = std::max(0., std::min(255., value));
		}
	}

	// Calculate how long took convolution
	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	// std::cout << image_rows << ", " << image_cols << ", " << duration.count() << std::endl;

	// Save gradients as images
	imwrite(base_name + "_x.png",
		Mat(image_rows, image_cols, CV_16SC1, Gx));
	imwrite(base_name + "_y.png",
		Mat(image_rows, image_cols, CV_16SC1, Gy));
	imwrite(base_name + "_edges.png",
		Mat(image_rows, image_cols, CV_8UC1, G));

	return EXIT_SUCCESS;
}
