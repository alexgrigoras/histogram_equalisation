# Histogram Equalization

## Description
Algorithm for histogram equalization implemented using CPU and GPU with CUDA.

## Implementation
- CPU version: implementation taken from [San Askaruly](https://gist.github.com/tuttelikz) / [Histogram Equalization OpenCV C++](https://gist.github.com/tuttelikz/bf20170368a8882c922afdf0bce399ed) and [opencv](https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e) 
- Naive GPU version using CUDA
- GPU version using CUDA with shared memory
- GPU version using CUDA with shared memory and optimizations
	- histogram kernel taken from [kevinzakka](https://github.com/kevinzakka) / [learn-cuda](https://github.com/kevinzakka/learn-cuda)
	
Further details are in the [Documentation](documentation.pdf).

## License
The applications are licensed under the MIT license.
