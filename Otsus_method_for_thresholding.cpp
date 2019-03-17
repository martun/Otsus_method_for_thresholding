#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

Mat make_greyscale(Mat& img) {
	Mat result(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3b intensity = img.at<Vec3b>(i, j);
			int avg = (intensity[0] + intensity[1] + intensity[2]) / 3;
			result.at<uint8_t>(i, j) = avg;
		}
	return result;
}

int find_otsus_threshold(const Mat& img) {
	int best_threshold;
	double max_variance = 0;

	// build the histogram.
	std::vector<int> histogram(256, 0);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			++histogram[img.at<uint8_t>(i, j)];
		}
	double weight0 = 0, mean0 = 0;
	double weight1 = 1, mean1 = 0;

	// Compute mean1
	for (int i = 0; i < 255; ++i) {
		mean1 += i * histogram[i];
	}
	mean1 /= img.rows * img.cols;

	double variance;
	// 254 to not divide by 0. At least one intensity will be in another set.
	for (int t = 1; t < 254; ++t) {
		double change = (double)histogram[t] / img.rows / img.cols;
		weight0 += change;
		weight1 -= change;
		mean0 += t * change;
		mean1 -= t * change;
		variance = weight0 * weight1 * 
			(mean0 / weight0 - mean1 / weight1) * (mean0 / weight0 - mean1 / weight1);
		if (variance > max_variance) {
			max_variance = variance;
			best_threshold = t;
		}
	}

	return best_threshold;
}
Mat apply_threshold(const Mat& img, int threshold) {
	Mat result(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			uint8_t pixel_value = img.at<uint8_t>(i, j);
			if (pixel_value > threshold) {
				result.at<uint8_t>(i, j) = 255;
			}
			else {
				result.at<uint8_t>(i, j) = 0;
			}
		}
	return result;
}

// A function to display multiple images in the same window,
// copied from the internet.
void display_multiple_images_in_one_window(string title, int nArgs, ...) {
	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row
	// h - Maximum number of images in a column
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying
	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 14) {
		printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
		return;
	}
	// Determine the size of the image,
	// and the number of rows/cols
	// from number of arguments
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC1);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		Mat img = va_arg(args, Mat);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img.empty()) {
			printf("Invalid arguments");
			return;
		}

		// Find the width and height of the image
		x = img.cols;
		y = img.rows;

		// Find whether height or width is greater in order to resize the image
		max = (x > y) ? x : y;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		// Set the image ROI to display the current image
		// Resize the input image and copy the it to the Single Big Image
		Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
		Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	// Create a new window, and show the Single Big Image
	namedWindow(title, 1);
	imwrite("ResultingImage.jpg", DispImage);
	imshow(title, DispImage);
	waitKey();

	// End the number of arguments
	va_end(args);
}
int main()
{
	Mat img1 = imread("Fish.png");
	Mat img2 = make_greyscale(img1);
	clock_t made_greyscale_time = clock();
	
	int threshold = find_otsus_threshold(img2);
	Mat img3 = apply_threshold(img2, threshold);

	display_multiple_images_in_one_window(
		"Steps of Basic region growing", 2,
		img2, img3);

	waitKey(0);
	return 0;
}