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
    for (long i = 0; i < dim; i++) printf("%d ", vect[i]);
}

void print_array(float* vect, int  dim)
{
    for (long i = 0; i < dim; i++) printf("%f ", vect[i]);
}

void display_histogram(int histogram[], const char* name) {
    int histogramWidth = 512;
    int histogramHeight = 400;
    int newHistogram[256];
    int binWidth;
    int maximumIntensity;

    for (int i = 0; i < 256; i++) newHistogram[i] = histogram[i];

    //creating "bins" for the range of 256 intensity values
    binWidth = cvRound((double)histogramWidth / 256);
    Mat histogramImage(histogramHeight, histogramWidth, CV_8UC1, Scalar(255, 255, 255));

    //finding maximum intensity level in the histogram
    maximumIntensity = newHistogram[0];
    for (int i = 1; i < 256; i++) {
        if (maximumIntensity < newHistogram[i]) maximumIntensity = newHistogram[i];
    }

    //normalizing histogram in terms of rows (y)
    for (int i = 0; i < 256; i++) newHistogram[i] = ((double)newHistogram[i] / maximumIntensity) * histogramImage.rows;

    //drawing the intensity level - line
    for (int i = 0; i < 256; i++) line(histogramImage, Point(binWidth * (i), histogramHeight), Point(binWidth * (i), histogramHeight - newHistogram[i]), Scalar(0, 0, 0), 1, 8, 0);

    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, histogramImage);
}

// Hillis & Steele Parallel Scan Algorithm
__global__ void cumHistKernelHS(int* d_out, int* d_in, int n)
{
    int idx = threadIdx.x;
    extern __shared__ int temp[];
    int pout = 0, pin = 1;

    temp[idx] = (idx > 0) ? d_in[idx - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        // swap double buffer indices
        pout = 1 - pout;
        pin = 1 - pout;
        if (idx >= offset) {
            temp[pout * n + idx] = temp[pin * n + idx - offset] + temp[pin * n + idx];  // changed line
        }
        else {
            temp[pout * n + idx] = temp[pin * n + idx];
        }
        __syncthreads();
    }
    d_out[idx] = temp[pout * n + idx];
}

// Shared memory using balanced trees (optimization)
__global__ void cumHistKernelBT(int* g_odata, int* g_idata, int n)
{
    extern __shared__ int temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
            
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) { temp[n - 1] = 0; } // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();

            if (thid < d)
            {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
                int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
            }
    }
    __syncthreads();
    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

__global__ void histogramKernel(int* d_out, int* d_in, long size)
{
    extern __shared__ unsigned int tempHist[];
    int tx = threadIdx.x;
    unsigned int idx = tx + blockIdx.x * blockDim.x;

    tempHist[tx] = 0;
    __syncthreads();
    if (idx < size) {
        atomicAdd(&(tempHist[d_in[idx]]), 1);       // add to private histogram
    }
    __syncthreads();
    atomicAdd(&(d_out[tx]), tempHist[tx]);          // contribute to global histogram.
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

__global__ void finalImageKernel(int* d_out, int* d_in)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = (uchar)(d_in[d_out[i]]);
}

int main()
{
    /*
    string image_str = "../images/img0";
    string extension = ".jpg";
    string img_name = image_str + extension;
    */
    Mat image = imread("D:/University/Master/Year 2/GPUP/Project/histogram_equalization/hist_equ_cuda/x64/Debug/img0.jpg", IMREAD_GRAYSCALE);

    int h = image.rows, w = image.cols;                             // image dimensions
    int* h_hist;
    int* h_image;
    float* h_PRk;
    int* h_cumHist;
    int* h_Sk;
    float* h_PSk;
    int* h_finalValues;
    int dim_hist = 256;
    int dim_image = h * w;                                          // image size
    float alpha = 255.0 / dim_image;
    cudaError_t cudaStatus;
    int numThreadsPerBlock = 256;                                   // define block size
    int numBlocks = dim_image / numThreadsPerBlock;
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocManaged(&h_hist, dim_hist * sizeof(int));
    cudaMallocManaged(&h_image, dim_image * sizeof(int));
    cudaMallocManaged(&h_PRk, dim_hist * sizeof(float));
    cudaMallocManaged(&h_cumHist, dim_hist * sizeof(int));
    cudaMallocManaged(&h_Sk, dim_hist * sizeof(int));
    cudaMallocManaged(&h_PSk, dim_hist * sizeof(float));
    cudaMallocManaged(&h_finalValues, dim_hist * sizeof(int));

    for (int i = 0; i < dim_hist; ++i) h_hist[i] = 0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            h_image[i * w + j] = image.at<uchar>(i, j);
        }
    }

    // Check CUDA device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaEventRecord(start, 0);  // Start global timers

    // ******************************************************************************************
    // Compute image histogram

    // launch kernel
    histogramKernel << < numBlocks, numThreadsPerBlock, dim_hist * sizeof(int) >> > (h_hist, h_image, dim_image);
    
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

    display_histogram(h_hist, "CUDA Histogram");

    // ******************************************************************************************
    // Compute Cumulative Histogram 

    cumHistKernelHS << < 1, dim_hist, 2 * dim_hist * sizeof(int) >> > (h_cumHist, h_hist, dim_hist);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[cumhist] addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // ******************************************************************************************
    // Probability distribution for intensity levels

    prkKernel << < 1, dim_hist >> > (h_PRk, h_hist, dim_image);
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

    // ******************************************************************************************
    // Scaling operation

    skKernel << < 1, dim_hist >> > (h_Sk, h_cumHist, alpha);
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
 
    // ******************************************************************************************
    // Mapping operation

    for (int i = 0; i < 256; i++) h_PSk[i] = 0.0;
   
    pskKernel << < 1, dim_hist >> > (h_PSk, h_Sk, h_PRk);
    
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
    
    // ******************************************************************************************
    // Rounding to get final values

    finalValuesKernel <<< 1, dim_hist >>> (h_finalValues, h_PSk);

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

    display_histogram(h_finalValues, "CUDA Equalized histogram");

    // ******************************************************************************************
    // Creating equalized image
    
    finalImageKernel << < numBlocks, numThreadsPerBlock >> > (h_image, h_Sk);

    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[final] addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            image.at<uchar>(i, j) = h_image[i * w + j];
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);    // cudaEventElapsedTime returns value in milliseconds.Resolution ~0.5ms
    printf("Execution time GPU: %f\n", elapsedTime);

Error:
    // Free device memory
    cudaFree(h_hist);
    cudaFree(h_image);
    cudaFree(h_PRk);
    cudaFree(h_cumHist);
    cudaFree(h_Sk);
    cudaFree(h_PSk);
    cudaFree(h_finalValues);
    // Free host memory
    // Destroy CUDA Event API Events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Display equalized image
    namedWindow("CUDA Equilized Image", WINDOW_NORMAL);
    imshow("CUDA Equilized Image", image);

    waitKey();

    return 0;
}