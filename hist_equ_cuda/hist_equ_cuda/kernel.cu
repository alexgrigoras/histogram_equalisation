#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace cv;
using namespace std;

void print_array(int* vect, int  dim)
{
    for (long i = 0; i < dim; i++) {
        printf("%d ", vect[i]);
    }
}

void print_array(float* vect, int  dim)
{
    for (long i = 0; i < dim; i++) {
        printf("%f ", vect[i]);
    }
}

void compute_cumulative_histogram(int histogram[], int cumulativeHistogram[]) {
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cumulativeHistogram[i] = histogram[i] + cumulativeHistogram[i - 1];
    }
}

void display_histogram(int histogram[], const char* name) {
    int histogramWidth = 512;
    int histogramHeight = 400;
    int newHistogram[256];

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

__global__ void histogramKernel(int* d_out, int* d_in) {
    int in = blockIdx.x * blockDim.x + threadIdx.x;
    int value = d_in[in];

    atomicAdd(&d_out[value], 1);
}
__global__ void prkKernel(float* d_out, int* d_in, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = (float)d_in[i] / size;
}
__global__ void skKernel(int* d_out, int* d_in, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = round((float)d_in[i] * alpha);
}
__global__ void pskKernel(float* d_out, int* d_in_a, float* d_in_b)
{
    int in = blockIdx.x * blockDim.x + threadIdx.x;
    int out = (int)d_in_a[in];

    atomicAdd(&d_out[out], d_in_b[in]);
}
__global__ void finalValuesKernel(int* d_out, float* d_in)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = round(d_in[i] * 255);
}
__global__ void finalImageKernel(int* d_out, int* d_in, int* d_img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = (uchar)(d_in[d_img[i]]);
}

int main()
{
    string image_str = "../images/img0";
    string extension = ".jpg";
    string img_name = image_str + extension;
    Mat image = imread(img_name, IMREAD_GRAYSCALE);
    int h = image.rows, w = image.cols;
    int* h_hist;                                                    // size of array
    int* h_image;                                                   // size of array
    int dim_hist = 256;
    int dim_image = h * w;                                          // 256K elements (1MB total)
    int* d_hist;                                                    // pointer to device memory
    int* d_image;                                                   // pointer to device memory
    cudaError_t cudaStatus;
    int numThreadsPerBlock = 256;                                   // define block size
    int numBlocks = dim_image / numThreadsPerBlock;
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_hist = new int[dim_hist];
    h_image = new int[dim_image];
    for (int i = 0; i < dim_hist; ++i)
    {
        h_hist[i] = 0;
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            h_image[i * w + j] = image.at<uchar>(i, j);
        }
    }

    // ******************************************************************************************

    // Compute image histogram

    cudaEventRecord(start, 0);  // Start global timers

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
    cudaStatus = cudaMemcpy(d_hist, h_hist, dim_hist * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
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

    display_histogram(h_hist, "CUDA Histogram");

    // ******************************************************************************************

    // Probability distribution for intensity levels

    float* h_PRk, * d_PRk;
    h_PRk = new float[dim_hist];
    cudaStatus = cudaMalloc((void**)&d_PRk, dim_hist * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    prkKernel << < 1, dim_hist >> > (d_PRk, d_hist, dim_image);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_PRk, d_PRk, dim_hist * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // ******************************************************************************************
        
    // Compute Cumulative Histogram 

    int* h_cumHist;
    h_cumHist = new int[dim_hist];
    compute_cumulative_histogram(h_hist, h_cumHist);

    // ******************************************************************************************

    // Scaling operation

    int* h_Sk, * d_Sk, * d_cumHist;
    h_Sk = new int[dim_hist];
    float alpha = 255.0 / dim_image;

    cudaStatus = cudaMalloc((void**)&d_Sk, dim_hist * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_cumHist, dim_hist * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_cumHist, h_cumHist, dim_hist * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    skKernel << < 1, dim_hist >> > (d_Sk, d_cumHist, alpha);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_Sk, d_Sk, dim_hist * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // ******************************************************************************************

    // Mapping operation

    float* h_PSk, * d_PSk;

    h_PSk = new float[dim_hist];
    for (int i = 0; i < 256; i++) {
        h_PSk[i] = 0.0;
    }
    cudaStatus = cudaMalloc((void**)&d_PSk, dim_hist * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_PSk, h_PSk, dim_hist * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    pskKernel << < 1, dim_hist >> > (d_PSk, d_Sk, d_PRk);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_PSk, d_PSk, dim_hist * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // ******************************************************************************************
    
    // Rounding to get final values

    int* h_finalValues, * d_finalValues;
    h_finalValues = new int[dim_hist];

    cudaStatus = cudaMalloc((void**)&d_finalValues, dim_hist * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    finalValuesKernel << < 1, dim_hist >> > (d_finalValues, d_PSk);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_finalValues, d_finalValues, dim_hist * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    display_histogram(h_finalValues, "CUDA Equalized histogram");

    // ******************************************************************************************

    // Creating equalized image

    int* d_finalImage;

    cudaStatus = cudaMalloc((void**)&d_finalImage, dim_image * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    finalImageKernel << < numBlocks, numThreadsPerBlock >> > (d_finalImage, d_Sk, d_image);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_image, d_finalImage, dim_image * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            image.at<uchar>(i, j) = h_image[i * w + j];
        }
    }

    /* CUDA Host / Device / Kernel Code ... */
    cudaEventRecord(stop, 0);                           
    cudaEventSynchronize(stop);                         
    cudaEventElapsedTime(&elapsedTime, start, stop);    // cudaEventElapsedTime returns value in milliseconds.Resolution ~0.5ms
    printf("Execution time GPU: %f\n", elapsedTime);

Error:
    // free device memory
    cudaFree(d_hist);
    cudaFree(d_image);
    cudaFree(d_PRk);
    cudaFree(d_cumHist);
    cudaFree(d_Sk);
    cudaFree(d_PSk);
    cudaFree(d_finalValues);
    cudaFree(d_finalImage);
    // free host memory
    std::free(h_hist);
    std::free(h_image);
    std::free(h_PRk);
    std::free(h_cumHist);
    std::free(h_Sk);
    std::free(h_PSk);
    std::free(h_finalValues);
    // Destroy CUDA Event API Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Display equalized image
    namedWindow("CUDA Equilized Image", WINDOW_NORMAL);
    imshow("CUDA Equilized Image", image);

    waitKey();

    return 0;
}