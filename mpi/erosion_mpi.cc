/*
 * Parallel Erosion
 * image process by opencv2, parallel by MPI
 * Author: huieric, Jinhui Zhu, ACM1501, HUST, China
 * Time: July 15th, 2018
 * 
 */
#include <malloc.h>
#include <iostream>
#include <string>
#include "mpi.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// threshold parameters
int threshold_value = 100;
int threshold_type = 0;
const int max_binary_value = 255;
// erosion parameter
const int element_rows = 3;
const int element_cols = 3;

const string data_path = "../../data/";
// set default data path
const int n = 512;
const int m = 512;

// function for process
void erode_center(const Mat& src, int center_x, int center_y, Mat& dst);

int main(int argc, char** argv) {
    int rank, size, avg;
    // Initialize MPI environment
    MPI::Init();
    // get current rank id
    rank = MPI::COMM_WORLD.Get_rank();
    // get total process number
    size = MPI::COMM_WORLD.Get_size();
    if(size != 1) avg = n / (size - 1);

    // if i am a main process
    if(rank == 0) {
        Mat src, src_gray, src_binary, erosion_dst;        
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
        erosion_dst.create(n, m, CV_8U);

        // if only main process, no worker process
        if(size == 1) {
            double tb, te;
            tb = MPI::Wtime();
            // sequentially erode the whole image
            for(int i=0; i<n; i++)
                for(int j=0; j<m; j++) {
                    erode_center(src_binary, j, i, erosion_dst);
                }
            te = MPI::Wtime();
            printf("openMPI cost time = %lf s\n", te-tb);
        }
        // if at least one worker process
        else {
            double tb, te;
            tb = MPI::Wtime();
            uchar* data = src_binary.data;
            // send the whole src image to each worker
            for(int i=0; i<size-1; i++)
                MPI::COMM_WORLD.Send(data, n*m, MPI::UNSIGNED_CHAR, i+1, 1);
            // receive results from each worker and combine them to get the whole eroded image
            for(int i=0; i<size-1; i++) {
                uchar* addr = erosion_dst.ptr<uchar>(i*avg);
                MPI::COMM_WORLD.Recv(addr, avg*m, MPI::UNSIGNED_CHAR, i+1, 2);
            }            
            te = MPI::Wtime();
            printf("openMPI cost time = %lf s\n", te-tb);
        }    
        imwrite("erosion_out.png", erosion_dst);    
    }
    // if i am a worker process
    if(size != 1 && rank != 0) {
        Mat src, dst;
        src.create(n, m, CV_8U);
        // receive the whole src image from main process
        MPI::COMM_WORLD.Recv(src.data, n*m, MPI::UNSIGNED_CHAR, 0, 1);
        dst.create(n, m, CV_8U);
        // erode lines that should done by me
        for(int i=avg*(rank-1); i<avg*rank; i++) {            
            for(int j=0; j<m; j++)
                erode_center(src, j, i, dst);
        }
        uchar* data = dst.ptr<uchar>(avg*(rank-1));          
        // send results to main process
        MPI::COMM_WORLD.Send(data, avg*m, MPI::UNSIGNED_CHAR, 0, 2);
        printf("my rankID = %d, receive line %d-%d of src image, send line %d-%d of eroded image\n", rank, 0, n-1, avg*(rank-1), avg*rank-1);
    }
    // finalize MPI environment
    MPI::Finalize();
    return 0;
}

void erode_center(const Mat& src, int center_x, int center_y, Mat& dst) {
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
        for(int i=left; i<=right; i++) {
            for(int j=top; j<=bottom; j++) {
                // printf("%d ", src_binary.at<uchar>(j, i));
                if(src.at<uchar>(j, i)==0) {
                    fit = false;
                    break;
                }                    
            }    
            // printf("\n");
        }                    
        retval = fit? 255 : 0;
    }
    uchar* data = dst.ptr<uchar>(center_y);
    data[center_x] = retval;    
}