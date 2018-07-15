// 相关 CUDA 库
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include<opencv2/opencv.hpp>
#include <time.h>
#include<iostream>

#define PARTS 4

using namespace cv;
using namespace std;

typedef struct st_range {
	int x1;
	int y1;
	int x2;
	int y2;
}StRange;

Mat srcImage, grayImage, binarygray;

const int N = 100;
const int BLOCK_data = 1;	// 块数
const int THREAD_data = 4;	// 各块中的线程数

static void * g_binary(void *range)		//二值化
{
	StRange rg = *(StRange *)range;
	binarygray = Mat::zeros(grayImage.rows, grayImage.cols, grayImage.type());
	for (int i = rg.x1; i < rg.x2; i++)
	{
		for (int j = rg.y1; j < rg.y2; j++)
		{
			if (grayImage.data[i*grayImage.step + j] > 128)
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
// 此函数在主机端调用，设备端执行。
__global__ static void g_dilation(unsigned char *imgData, unsigned char *result, int rows, int cols)  //腐蚀
{
	StRange srcRange, rg, ranges[PARTS];
	srcRange = { 0, 0, rows, cols };
	//切分图像
	ranges[0] = { 0, 0, srcRange.x2 / 4, srcRange.y2 };
	ranges[1] = { srcRange.x2 / 4, 0, srcRange.x2 / 2, srcRange.y2 };
	ranges[2] = { srcRange.x2 / 2 ,0 , 3 * srcRange.x2 / 4, srcRange.y2 };
	ranges[3] = { 3 * srcRange.x2 / 4, 0 , srcRange.x2, srcRange.y2 };
	for (int tid = 0; tid < 4; tid++)
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
__global__ static void g_erosion(unsigned char *imgData, unsigned char *result, int rows, int cols)  //膨胀
{
	StRange srcRange, rg, ranges[PARTS];
	srcRange = { 0, 0, rows, cols };
	//切分图像
	ranges[0] = { 0, 0, srcRange.x2 / 4, srcRange.y2 };
	ranges[1] = { srcRange.x2 / 4, 0, srcRange.x2 / 2, srcRange.y2 };
	ranges[2] = { srcRange.x2 / 2 ,0 , 3 * srcRange.x2 / 4, srcRange.y2 };
	ranges[3] = { 3 * srcRange.x2 / 4, 0 , srcRange.x2, srcRange.y2 };
	for (int tid = 0; tid < 4; tid++)
	{
		if (tid == threadIdx.x)
		{
			printf("thid:%d\n", tid);
			rg = ranges[tid];
			//printf("x1: %d, y1: %d\nx2: %d, y2:%d\n", ranges[tid].x1, ranges[tid].y1, ranges[tid].x2, ranges[tid].y2);
			for (int i = rg.x1; i < rg.x2; i++)
			{
				for (int j = rg.y1; j < rg.y2; j++)
				{
					if (imgData[(i - 1)*cols + j] == 0 || imgData[(i - 1)*cols + j - 1] == 0 || imgData[i*cols + j + 1] == 0)
					{
						result[i*cols + j] = 0;
					}
					else
					{
						result[i*cols + j] = 255;
					}
				}
			}
			printf("Over thread%d\n\n", tid);
		}
	}
}
// CUDA初始化函数
bool InitCUDA()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);	// 获取显示设备数
	if (deviceCount == 0)
	{
		cout << "找不到设备" << endl;
		return EXIT_FAILURE;
	}
	int i;
	for (i = 0; i<deviceCount; i++)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) // 获取设备属性
		{
			if (prop.major >= 1) //cuda计算能力
			{
				break;
			}
		}
	}
	if (i == deviceCount)
	{
		cout << "找不到支持 CUDA 计算的设备" << endl;
		return EXIT_FAILURE;
	}
	cudaSetDevice(i); // 选定使用的显示设备
	return EXIT_SUCCESS;
}

int main()
{
	if (InitCUDA()) // 初始化 CUDA 编译环境
		return EXIT_FAILURE;
	cout << "成功建立 CUDA 计算环境" << endl << endl;

	clock_t begin, end;
	double cost;
	cout << "\n\n本程序涉及到：" << "腐蚀（erosion）、膨胀（dilation)。\n\n";
	
	system("color 3f");
	srcImage = imread("D://2.jpg");
	imshow("原图", srcImage);
	cvtColor(srcImage, grayImage, CV_RGB2GRAY);		//RGB图像转换为灰度图
	StRange srcRange = { 0, 0, srcImage.rows, srcImage.cols };
	g_binary(&srcRange);		//灰度图二值化处理
	imshow("binarygray", binarygray);

	// 传递参数
	unsigned char *img, *result1, *result2;
	int arraySize = sizeof(unsigned char)*srcImage.cols * srcImage.rows;

	//开始记录
	begin = clock();
	// 在显存中为计算对象开辟空间
	cudaMalloc((void**)&img, arraySize);
	// 在显存中为结果对象开辟空间
	cudaMalloc((void**)&result1, arraySize);
	cudaMalloc((void**)&result2, arraySize);

	// 将数据传输进显存
	cudaMemcpy(img, binarygray.data, arraySize, cudaMemcpyHostToDevice);

	// 调用 kernel 函数 - 此函数可以根据显存地址以及自身的块号，线程号处理数据。
	g_erosion << <BLOCK_data, THREAD_data, 0 >> > (img, result1, srcImage.rows, srcImage.cols);
	g_dilation << <BLOCK_data, THREAD_data, 0 >> > (img, result2, srcImage.rows, srcImage.cols);
	// 在内存中为计算对象开辟空间
	unsigned char * resData1 = new unsigned char[srcImage.rows * srcImage.cols];
	unsigned char * resData2 = new unsigned char[srcImage.rows * srcImage.cols];
	// 从显存获取处理的结果
	cudaMemcpy(resData1, result1, arraySize, cudaMemcpyDeviceToHost);
	cudaMemcpy(resData2, result2, arraySize, cudaMemcpyDeviceToHost);

	Mat erosion(srcImage.rows, srcImage.cols, CV_8UC1, resData1);
	Mat dilation(srcImage.rows, srcImage.cols, CV_8UC1, resData2);
	imshow("dilation", dilation);
	imshow("erosion", erosion);

	// 释放显存
	cudaFree(img);
	cudaFree(result1);
	cudaFree(result2);

	end = clock();
	cost = (double)(end - begin);
	printf("Time cost is: %lf ms", cost);
	waitKey(0);
	return 0;
}