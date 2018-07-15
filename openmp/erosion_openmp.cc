/*
 * Parallel Erosion
 * image process by opencv2, parallel by openMP
 * Author: huieric, Jinhui Zhu, ACM1501, HUST, China
 * Time: July 15th, 2018
 * 
 */
#include <malloc.h>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
// set thread number
#define THREAD 512

using namespace std;
using namespace cv;

// global Mat object for thread input and output
Mat src, src_gray, src_binary, erosion_dst;
int n, m, avg;
// threshold parameters
int threshold_value = 100;
int threshold_type = 0;
const int max_binary_value = 255;
// erosion parameter
const int element_rows = 3;
const int element_cols = 3;
// set default data path
const string data_path = "../../data/";

// function for threads
void* erode_line(void* arg);
uchar erode_center(int center_x, int center_y);

int main(int argc, char** argv) {
    // parse command line parameters
    CommandLineParser parser(argc, argv, "{@input | chicky_512.png | input image}");
    // read image from path provided by command line suffixed with default data path
    src = imread(data_path+parser.get<String>("@input"), IMREAD_COLOR);
    if(src.empty()) {
        printf("Could not open or find the image!\n\n");
        printf("Usage: %s <Input image>\n", argv[0]);
        return -1;
    }
    // convert src image to grayscale image
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    // threshold graysale image to generate binary image
    threshold(src_gray, src_binary, threshold_value, max_binary_value, threshold_type);
    imwrite("src_gray.png", src_gray);
    imwrite("src_binary.png", src_binary);

    n=src_binary.rows;
    m=src_binary.cols;
    // evenly partition point of image by thread number
    avg = n / THREAD;
    printf("The dimension of binary image to erode: %d x %d\n", n, m);
    erosion_dst.create(n, m, CV_8U);
    time_t t_start, t_end;
    
    t_start = clock();
    // compiler will automatically divide for loop into multiple threads to parallel execute
    #pragma omp parallel for
    for(int i=0; i<THREAD; i++) {
        void* arg = malloc(2*sizeof(int));
        ((int*)arg)[0] = i*avg;
        ((int*)arg)[1] = avg;
        erode_line(arg);
    }
    // generate the final image
    imwrite("erosion_out.png", erosion_dst);
    printf("picture generation... done\n");
    t_end = clock();
    printf("openMP cost time: %f s\n", double(t_end-t_start)/CLOCKS_PER_SEC);
    return 0;
}

void* erode_line(void* arg) {
    int start = ((int*)arg)[0];
    int t_id = start / avg;
    int lines = ((int*)arg)[1];
    if(t_id + 1 == THREAD) lines = n - start;
    free(arg);
    for(int i=0; i<lines; i++) {
        int cur = start+i;
        for(int j=0; j<m; j++) {
            uchar* data = erosion_dst.ptr<uchar>(cur);
            data[j] = erode_center(j, cur);
        }
    }
    printf("pthread %d: erosion from line %d to %d... done\n", t_id, start, start+lines-1);
    return 0;
}

uchar erode_center(int center_x, int center_y) {
    int left = center_x-(element_cols-1)/2;
    int right = center_x+(element_cols-1)/2;
    int top = center_y-(element_rows-1)/2;
    int bottom = center_y+(element_rows-1)/2;
    uchar retval;
    if(left<0 || right>=m || top<0 || bottom>=n) {
        retval = 0;
    }
    else {
        bool fit = true;
        for(int j=top; j<=bottom; j++) {
            for(int i=left; i<=right; i++) {
                // printf("%d ", src_binary.at<uchar>(j, i));
                assert(0<=i && i<m);
                assert(0<=j && j<n);
                if(src_binary.at<uchar>(j, i)==0) {
                    fit = false;
                    break;
                }                    
            }    
            // printf("\n");
        }                    
        retval = fit? 255 : 0;
    }
    return retval;
}