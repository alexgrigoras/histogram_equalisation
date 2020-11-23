
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void print_array(int* vect, int  dim)
{
    for (long i = 0; i < dim; i++) {
        printf("%d ", vect[i]);
    }
}

void display_histogram(int histogram[], const char* name) {
    int newHistogram[256];
    for (int i = 0; i < 256; i++) {
        newHistogram[i] = histogram[i];
    }
    //histogram size
    int histogramWidth = 512;
    int histogramHeight = 400;
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

// compute histogram kernel
__global__ void histogramKernel(int* d_out, int* d_in) {

    int inOffset = blockDim.x * blockIdx.x;
    int in = inOffset + threadIdx.x;
    int value = d_in[in];

    atomicAdd(&d_out[value], 1);
}

int main()
{
    Mat image = imread("../images/img.jpg", IMREAD_GRAYSCALE);
    imshow("Original Image", image);

    // pointer to host memory
    int h = image.rows, w = image.cols;
    int* h_hist;                       // size of array
    int* h_image;                           // size of array
    int dim_hist = 256;
    int dim_image = h*w;                    // 256K elements (1MB total)
    int* d_hist;                            // pointer to device memory
    int* d_image;                           // pointer to device memory
    cudaError_t cudaStatus;
    int numThreadsPerBlock = 25;          // define block size
    // compute number of blocks needed based on
    // array size and desired block size
    int numBlocks = dim_image / numThreadsPerBlock;
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);  // allocate host and device memory

    h_hist = new int[dim_hist];
    h_image = new int[dim_image];

    // Initialize input array on host
    for (int i = 0; i < dim_hist; ++i)
    {
        h_hist[i] = 0;
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            h_image[i * w + j] = image.at<uchar>(i, j);
        }
    }

    // Copy host array to device array
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_image, dim_image * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_hist, dim_hist * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_image, h_image, dim_image * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // launch kernel
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    histogramKernel <<< dimGrid, dimBlock >>> (d_hist, d_image);
    // block until the device has completed
    cudaThreadSynchronize();

    // device to host copy
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(h_hist, d_hist, dim_hist * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    printf("CUDA Histogram:");
    print_array(h_hist, dim_hist);
    printf("\n");

    display_histogram(h_hist, "CUDA Histogram");

Error:
    // free device memory
    cudaFree(d_hist);
    cudaFree(d_image);

    // free host memory
    std::free(h_hist);
    std::free(h_image);

    waitKey();

    return 0;
}