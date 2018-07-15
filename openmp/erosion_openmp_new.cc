#include <malloc.h>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat src, src_gray, src_binary, erosion_dst;
int n, m;
int threshold_value = 100;
int threshold_type = 0;
const int max_binary_value = 255;
const int element_rows = 3;
const int element_cols = 3;
const string data_path = "../../data/";

void* erode_center(void* arg);

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, "{@input | chicky_512.png | input image}");
    src = imread(data_path+parser.get<String>("@input"), IMREAD_COLOR);
    if(src.empty()) {
        printf("Could not open or find the image!\n\n");
        printf("Usage: %s <Input image>\n", argv[0]);
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    threshold(src_gray, src_binary, threshold_value, max_binary_value, threshold_type);
    imwrite("src_gray.png", src_gray);
    imwrite("src_binary.png", src_binary);

    n=src_binary.rows;
    m=src_binary.cols;
    printf("The dimension of binary image to erode: %d x %d\n", n, m);
    void* retval;
    erosion_dst.create(n, m, CV_8U);
    time_t t_start, t_end;
    
    t_start = clock();
    #pragma omp parallel for num_threads(4)
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++) {
            uchar* data = erosion_dst.ptr<uchar>(i);
            void* arg = malloc(2*sizeof(int));
            ((int*)arg)[0] = i;
            ((int*)arg)[1] = j;
            retval = erode_center(arg);
            data[j] = *(uchar*)retval;
            free(retval);
        }
	t_end = clock();
    imwrite("erosion_out.png", erosion_dst);
    printf("picture generation... done\n");
    printf("openMP cost time: %f s\n", double(t_end-t_start)/CLOCKS_PER_SEC);
    return 0;
}

void* erode_center(void* arg) {
    int center_y = ((int*)arg)[0];
    int center_x = ((int*)arg)[1];
    int left = center_x-(element_cols-1)/2;
    int right = center_x+(element_cols-1)/2;
    int top = center_y-(element_rows-1)/2;
    int bottom = center_y+(element_rows-1)/2;
    void* retval = malloc(sizeof(uchar));
    if(left<0 || right>=m || top<0 || bottom>=n) {
        *(uchar*)retval = 0;
    }
    else {
        bool fit = true;
        for(int i=left; i<=right; i++) {
            for(int j=top; j<=bottom; j++) {
                // printf("%d ", src_binary.at<uchar>(j, i));
                if(src_binary.at<uchar>(j, i)==0) {
                    fit = false;
                    break;
                }                    
            }    
            // printf("\n");
        }                    
        *(uchar*)retval = fit? 255 : 0;
    }
    free(arg);
    return retval;
}