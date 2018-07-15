/*
 * Parallel BFS
 * parallel by openMP
 * Author: huieric, Jinhui Zhu, ACM1501, HUST, China
 * Time: July 15th, 2018
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <queue>
#include "omp.h"
// set node number
#define N 1024

using namespace std;

// adjacency matrix
int T[N][N] = {0};

// 75% two different nodes have an edge
int randInt(int row, int col) {
    if(row == col)
        return 0;
    int r = rand() % 4; 
    if(r == 0)
        return 0;
    else 
        return 1;
}

// parallel init a tree
void initTree() {
    #pragma omp parallel for
    for(int i=0; i<N; i++)
        #pragma omp parallel for
        for(int j=0; j<N; j++) {
            int r = randInt(i, j);
            T[i][j] = r;
            T[j][i] = r;
        }    
}

void BFS(int start) {
    // label all nodes as unvisited
    bool visited[N] = {false};
    queue<int> q;
    // push start node into queue
    q.push(start);
    // label start as visited
    visited[start] = true;
    int counter = 0;
    while(!q.empty()) {
        int head = q.front();
        q.pop();
        // parallel visit each child tree
        #pragma omp parallel for
        for(int i=0; i<N; i++) {
            if(T[head][i] == 1 && !visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
        printf("%d -> ", head+1);
        counter++;
        if(counter == 10) {
            counter %= 10;
            printf("\n");
        }
    }
    printf("\n");
}

int main() {
    time_t start, end;
    initTree();
    start = clock();
    BFS(0);
    end = clock();
    printf("Parallel BFS (openmp) on tree with %d nodes spent %lf s\n", N, double(end-start)/CLOCKS_PER_SEC);
    return 0;
}