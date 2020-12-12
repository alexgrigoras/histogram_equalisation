#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono> 
#include <math.h>
#include <string.h>

using namespace std::chrono;
using namespace std;
using namespace cv;

void print_array(int* vect, int  dim)
{
    for (long i = 0; i < dim; i++) printf("%d ", vect[i]);
}

void print_array(float* vect, int  dim)
{
    for (long i = 0; i < dim; i++) printf("%f ", vect[i]);
}

void compute_histogram(Mat image, int histogram[]) {
    // initialize all intensity values to 0
    for (int i = 0; i < 256; i++) histogram[i] = 0;

    // calculate the number of pixels for each intensity value
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            histogram[(int)image.at<uchar>(y, x)]++;
        }
    }
}

void compute_cumulative_histogram(int histogram[], int cumulativeHistogram[]) {
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 256; i++) cumulativeHistogram[i] = histogram[i] + cumulativeHistogram[i - 1];
}

void display_histogram(int histogram[], const char* name) {
    int newHistogram[256];
    int histogramWidth = 512;
    int histogramHeight = 400;

    for (int i = 0; i < 256; i++) {
        newHistogram[i] = histogram[i];
    }

    //creating "bins" for the range of 256 intensity values
    int binWidth = cvRound((double)histogramWidth / 256);
    Mat histogramImage(histogramHeight, histogramWidth, CV_8UC1, Scalar(255, 255, 255));
    
    //finding maximum intensity level in the histogram
    int maximumIntensity = newHistogram[0];
    for (int i = 1; i < 256; i++) {
        if (maximumIntensity < newHistogram[i]) {
            maximumIntensity = newHistogram[i];
        }
    }

    //normalizing histogram in terms of rows (y)
    for (int i = 0; i < 256; i++) {
        newHistogram[i] = ((double)newHistogram[i] / maximumIntensity) * histogramImage.rows;
    }

    //drawing the intensity level - line
    for (int i = 0; i < 256; i++) {
        line(histogramImage, Point(binWidth * (i), histogramHeight), Point(binWidth * (i), histogramHeight - newHistogram[i]), Scalar(0, 0, 0), 1, 8, 0);
    }

    // display
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histogramImage);
}

int main()
{
    string image_str = "../images/img0";
    string extension = ".jpg";
    string img_name = image_str + extension;
    Mat image = imread(img_name, IMREAD_GRAYSCALE);
    Mat equalized_image = image.clone();
    int size = image.rows * image.cols;
    float alpha = 255.0 / size;
    int histogram[256];
    int cumulativeHistogram[256];
    float PRk[256];
    int Sk[256];
    float PSk[256];
    int finalValues[256];
    int i, x, y;

    auto start = high_resolution_clock::now();

    compute_histogram(image, histogram);

    // Probability distribution for intensity levels
    for (i = 0; i < 256; i++) PRk[i] = (float)histogram[i] / size;

    compute_cumulative_histogram(histogram, cumulativeHistogram);

    // Scaling operation
    for (i = 0; i < 256; i++) Sk[i] = cvRound((float)cumulativeHistogram[i] * alpha);

    // Mapping operation
    for (i = 0; i < 256; i++) PSk[i] = 0;
    for (i = 0; i < 256; i++) PSk[Sk[i]] += PRk[i];

    // Rounding to get final values
    for (i = 0; i < 256; i++) finalValues[i] = cvRound(PSk[i] * 255);

    // Creating equalized image
    for (y = 0; y < image.rows; y++)
        for (x = 0; x < image.cols; x++)
            equalized_image.at<uchar>(y, x) = saturate_cast<uchar>(Sk[image.at<uchar>(y, x)]);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << "Execution time CPU: " << (float)duration.count() / 1000000 << endl;

    display_histogram(histogram, "Original Histogram");
    namedWindow("Equalized Image", WINDOW_NORMAL);
    imshow("Equalized Image", equalized_image);
    display_histogram(finalValues, "Equalized Histogram");

    waitKey();
    return 0;
}