#include <pthread.h>
#include <malloc.h>
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

Mat src, erosion_dst;

void* f(void* arg);

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, "{@input | img/img1.png | input image}");
    src = imread(parser.get<String>("@input"), IMREAD_COLOR);
    if(src.empty()) {
        printf("Could not open or find the image!\n\n");
        printf("Usage: %s <Input image>\n", argv[0]);
        return -1;
    }

    int n=10, m=10;
    pthread_t p[n][m];
    void* retval[n][m];
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++) {
            void* arg = malloc(2*sizeof(int));
            ((int*)arg)[0] = i;
            ((int*)arg)[1] = j;
            pthread_create(&p[i][j], 0, f, arg);
        }
    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++) {
            pthread_join(p[i][j], &retval[i][j]);
        }
    // test return value
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            void* addr = retval[i][j];
            if(addr) {
                printf("%d ", *(int*)addr);
                free(addr);
            }                            
        }
        putchar('\n');
    }
    return 0;
}

void* f(void* arg) {
    int i = ((int*)arg)[0];
    int j = ((int*)arg)[1];
    // check if parallel
    printf("pthread (%d, %d): line %d column %d\n", i, j, i, j);
    void* retval = malloc(sizeof(int));
    *(int*)retval = (i+j)/2;
    free(arg);
    return retval;
}