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

	// Implement the convolution algorithm using NEON intrinsics

	for (int y = 1; y < image_rows - 1; y++) {
		for (int x = 1; x < image_cols - 1;
		     x += 8) { // Process 8 pixels at a time

			// Load rows for 3x3 neighborhood
			int16x8_t top = vld1q_s16(
				&src_data[(y - 1) * image_cols + (x - 1)]);
			int16x8_t mid =
				vld1q_s16(&src_data[y * image_cols + (x - 1)]);
			int16x8_t bot = vld1q_s16(
				&src_data[(y + 1) * image_cols + (x - 1)]);

			int16x8_t grad_x_top =
				vmlaq_s16(vmlaq_s16(vmulq_s16(top, kx_row1),
						    mid, kx_row2),
					  bot, kx_row3);
			int16x8_t grad_x_bot =
				vmlaq_s16(vmlaq_s16(vmulq_s16(bot, kx_row1),
						    mid, kx_row2),
					  top, kx_row3);
			int16x8_t gx = vsubq_s16(grad_x_top, grad_x_bot);

			int16x8_t grad_y_top =
				vmlaq_s16(vmlaq_s16(vmulq_s16(top, ky_row1),
						    mid, ky_row2),
					  bot, ky_row3);
			int16x8_t grad_y_bot =
				vmlaq_s16(vmlaq_s16(vmulq_s16(bot, ky_row1),
						    mid, ky_row2),
					  top, ky_row3);
			int16x8_t gy = vsubq_s16(grad_y_top, grad_y_bot);

			vst1q_s16(&Gx[y * cols + x], gx);
			vst1q_s16(&Gy[y * cols + x], gy);
		}
	}

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
