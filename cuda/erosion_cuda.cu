/*
 * Parallel Erosion
 * image process by opencv2, parallel by MPI
 * Author: huieric, Jinhui Zhu, ACM1501, HUST, China
 * Time: July 15th, 2018
 * 
 */
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

// set thread number
#define PARTS 1

using namespace cv;
using namespace std;

// define StRange to describe a specific block of image
typedef struct st_range {
	int x1;
	int y1;
	int x2;
	int y2;
}StRange;

Mat srcImage, grayImage, binarygray;

const int N = 100;
const int BLOCK_data = 1;	// block size
const int THREAD_data = 4;	// thread number in each block

// convert grayscale image to binary image
static void * g_binary(void *range)
{
	StRange rg = *(StRange *)range;
	binarygray = Mat::zeros(grayImage.rows, grayImage.cols, grayImage.type());
	for (int i = rg.x1; i < rg.x2; i++)
	{
		for (int j = rg.y1; j < rg.y2; j++)
		{
			if (grayImage.data[i*grayImage.step + j] > 100)
			{
				binarygray.data[i*binarygray.step + j] = 255;		//white
			}
			else
			{
				binarygray.data[i*binarygray.step + j] = 0;			//black
			}
		}
	}
	return NULL;
}
// dilation and erosion called by host and executed by device
__global__ static void g_dilation(unsigned char *imgData, unsigned char *result, int rows, int cols)  // dilation
{
	StRange srcRange, rg, ranges[PARTS];
	srcRange = { 0, 0, rows, cols };
	// partition image
	// ranges[0] = { 0, 0, srcRange.x2 / 4, srcRange.y2 };
	// ranges[1] = { srcRange.x2 / 4, 0, srcRange.x2 / 2, srcRange.y2 };
	// ranges[2] = { srcRange.x2 / 2 ,0 , 3 * srcRange.x2 / 4, srcRange.y2 };
	// ranges[3] = { 3 * srcRange.x2 / 4, 0 , srcRange.x2, srcRange.y2 };
	for (int tid = 0; tid < PARTS; tid++)
	{
		if (tid == threadIdx.x)
		{
			//printf("thid:%d\n", tid);
			rg = ranges[tid];
			//printf("x1: %d, y1: %d\nx2: %d, y2:%d\n", ranges[tid].x1, ranges[tid].y1, ranges[tid].x2, ranges[tid].y2);
			for (int i = rg.x1; i < rg.x2; i++)
			{
				for (int j = rg.y1; j < rg.y2; j++)
				{
					if (imgData[(i - 1)*cols + j] + imgData[(i - 1)*cols + j + 1] + imgData[i*cols + j + 1] == 0)
					{
						result[i*cols + j] = 0;
					}
					else
					{
						result[i*cols + j] = 255;
					}
				}
			}
			//printf("Over thread%d\n\n", tid);
		}
	}
}
__global__ static void g_erosion(unsigned char *imgData, unsigned char *result, int rows, int cols)  //    
{
	StRange srcRange, rg, ranges[PARTS];
	srcRange = { 0, 0, rows, cols };
	// ranges[0] = { 0, 0, srcRange.x2 / 4, srcRange.y2 };
	// ranges[1] = { srcRange.x2 / 4, 0, srcRange.x2 / 2, srcRange.y2 };
	// ranges[2] = { srcRange.x2 / 2 ,0 , 3 * srcRange.x2 / 4, srcRange.y2 };
	// ranges[3] = { 3 * srcRange.x2 / 4, 0 , srcRange.x2, srcRange.y2 };
	// evenly partition binary image
	const int avg = srcRange.x2 / PARTS;	
	for (int i = 0; i < PARTS; i++)
		ranges[i] = { i * avg, 0, (i + 1) * avg, srcRange.y2 };
	ranges[PARTS - 1].x2 = srcRange.x2;
	// for each thread
	for (int tid = 0; tid < PARTS; tid++)
	{
		// if i am current thread
		if (tid == threadIdx.x)
		{
			// get my block range in the image
			rg = ranges[tid];	
			// point in the block
			for (int i = rg.x1; i < rg.x2; i++)
			{
				for (int j = rg.y1; j < rg.y2; j++)
				{
					// element 3x3, all 1
					// if not fit
					if (imgData[(i - 1)*cols + j] == 0
						|| imgData[(i - 1)*cols + j - 1] == 0
						|| imgData[(i - 1)*cols + j + 1] == 0
						|| imgData[i*cols + j + 1] == 0
						|| imgData[i*cols + j] == 0
						|| imgData[i*cols + j - 1] == 0
						|| imgData[(i + 1)*cols + j + 1] == 0
						|| imgData[(i + 1)*cols + j] == 0
						|| imgData[(i + 1)*cols + j - 1] == 0)
					{
						result[i*cols + j] = 0;
					}
					// if fit
					else
					{
						result[i*cols + j] = 255;
					}
				}
			}
			printf("CUDA thread %d: erosion from line %d to %d... done\n", tid, rg.x1, rg.x2-1);
		}
	}
}
	
// CUDA init
bool InitCUDA()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);	// get devices number
	if (deviceCount == 0)
	{
		cout << "Can not find a device" << endl;
		return EXIT_FAILURE;
	}
	int i;
	for (i = 0; i<deviceCount; i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) // get device attribute
		{
			if (prop.major >= 1) // cuda computing ability
			{
				break;
			}
		}
	}
	if (i == deviceCount)
	{
		cout << "Can not find a device supporting cuda computation" << endl;
		return EXIT_FAILURE;
	}
	cudaSetDevice(i); // select a device
	return EXIT_SUCCESS;
}

int main()
{
	if (InitCUDA()) // init cuda environment
		return EXIT_FAILURE;
	cout << "Initialize CUDA computing environment... done" << endl;

	clock_t begin, end;
	double cost;
	
	system("color 3f");
	srcImage = imread("D://2.jpg");
	cvtColor(srcImage, grayImage, CV_RGB2GRAY);		// convert RPG image to grayscale image
	StRange srcRange = { 0, 0, srcImage.rows, srcImage.cols };
	printf("The dimension of binary image: %d x %d\n", srcImage.rows, srcImage.cols);
	g_binary(&srcRange);		// convert grayscale image to binary image
	// imshow("binarygray", binarygray);

	// pass paramenters
	unsigned char *img, *result1, *result2;
	int arraySize = sizeof(unsigned char)*srcImage.cols * srcImage.rows;

	begin = clock();
	// create space for src in the device memory
	cudaMalloc((void**)&img, arraySize);
	// create space for result in the device memory
	cudaMalloc((void**)&result1, arraySize);
	cudaMalloc((void**)&result2, arraySize);

	// pass data into device memory
	cudaMemcpy(img, binarygray.data, arraySize, cudaMemcpyHostToDevice);

	// call kernel function which can process data by block number, thread id
	g_erosion << <BLOCK_data, THREAD_data, 0 >> > (img, result1, srcImage.rows, srcImage.cols);
	g_dilation << <BLOCK_data, THREAD_data, 0 >> > (img, result2, srcImage.rows, srcImage.cols);
	// create space for result in the memory
	unsigned char * resData1 = new unsigned char[srcImage.rows * srcImage.cols];
	unsigned char * resData2 = new unsigned char[srcImage.rows * srcImage.cols];
	// get result from the device memory
	cudaMemcpy(resData1, result1, arraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(resData2, result2, arraySize, cudaMemcpyDeviceToHost);

	Mat erosion(srcImage.rows, srcImage.cols, CV_8UC1, resData1);
	Mat dilation(srcImage.rows, srcImage.cols, CV_8UC1, resData2);
	// imshow("dilation", dilation);
	// imshow("erosion", erosion);

	// release the device memory
	cudaFree(img);
	cudaFree(result1);
	cudaFree(result2);

	end = clock();
	printf("CUDA cost time = %lf s\n", (double)(end - begin)/CLOCKS_PER_SEC);
	waitKey(0);
	return 0;
}