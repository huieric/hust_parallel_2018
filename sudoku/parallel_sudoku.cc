/*
 * Parallel Sudoku
 * parallel by pthread
 * Author: huieric, Jinhui Zhu, ACM1501, HUST, China
 * Time: July 15th, 2018
 * 
 */
#include <time.h>
#include <stdlib.h>
#include <iostream>  
#include <pthread.h>
#include <stdio.h>
using namespace std;

// create Sudoku matrix
int num[9][9] = {
	3,0,2,7,0,0,0,0,9,
	0,0,8,0,0,0,0,4,5,
	0,0,4,0,0,1,3,0,0,
	0,0,0,0,5,9,0,0,0,
	0,9,0,0,3,0,0,6,0,
	0,0,0,2,6,0,0,0,0,
	0,0,1,4,0,0,2,0,0,
	2,6,0,0,0,0,1,0,0,
	4,0,0,0,0,2,5,0,3

};

int sum = 0;
typedef struct st_para {
	int n;		// position
	int data[9][9];	// result array
}Parameter;

void Output(int data[9][9]); // output answers
bool Check(int data[9][9], int n, int key); // check if is an answer
void * mythread(void * arg); // worker thread to find an answer
int DFS(int data[9][9], int n, int &count);

int main() {
	int count = 0;
	time_t begin, end;
	begin = clock();
	cout << "需要求解的数独问题是：" << endl;
	Output(num);
	cout << "【begin】" << endl;
	Parameter para;
	para.n = 0;
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++) {
			para.data[i][j] = num[i][j];
		}
	}
	pthread_t pid;
	// create thread: get pid, set function address, set actual parameters
	pthread_create(&pid, NULL, mythread, (void *)&para);
	// suspend parent process and wait for threads running 
	pthread_join(pid, NULL);
	cout << "【end】" << endl;
	end = clock();
	cout << "方案总数：" << sum << endl;
	printf("用时：%lf s\n", double(end - begin)/CLOCKS_PER_SEC);
	return 0;
}

void Output(int num[9][9])
{	cout << endl;
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			cout << num[i][j] << " ";
			if (j % 3 == 2)
				cout << "   ";
		}
		cout << endl;
		if (i % 3 == 2)
			cout << endl;
	}
}

/* 判断key填入n时是否满足条件 */
bool Check(int data[9][9], int n, int key) {
	for (int i = 0; i < 9; i++) {
		if (data[n / 9][i] == key) /* 判断n所在横列是否合法 */
			return false;
		if (data[i][n % 9] == key) /* 判断n所在竖列是否合法 */
			return false;
	}
	int x = n / 9 / 3 * 3;	/* x为n所在的小九宫格左顶点竖坐标 */
	int y = n % 9 / 3 * 3;	/* y为n所在的小九宫格左顶点横坐标 */
	for (int i = x; i < x + 3; i++) {	/* 判断n所在的小九宫格是否合法 */
		for (int j = y; j < y + 3; j++) {
			if (data[i][j] == key) return false;
		}
	}
	return true;
}

void * mythread(void *para) 
{
	Parameter * para1 = (Parameter *)para;   //传递的参数
	if (para1->n == 81) /* 递归出口，数组填满，找到一个解法 */
		return NULL;
	if (para1->data[para1->n / 9][para1->n % 9] != 0) /* 当前位不为空时跳过 */
	{
		Parameter * para2 = new Parameter;
		para2->n = para1->n + 1;
		for (int i = 0; i < 9; i++) 
			for (int j = 0; j < 9; j++) 
				para2->data[i][j] = para1->data[i][j];
		if (para1->n < 2) 
		{
			pthread_t pid;
			pthread_create(&pid, NULL, mythread, (void *)para2);
			pthread_join(pid, NULL);
		}
		else 
		{
			int count = 0;
			DFS(para2->data, para2->n, count);
		}
	}
	else {
		if (para1->n < 2) {
			pthread_t pid[10];
			for (int i = 1; i <= 9; i++) {
				if (Check(para1->data, para1->n, i) == true) {
					Parameter * para2 = new Parameter;
					para2->n = para1->n + 1;
					for (int i = 0; i < 9; i++) {
						for (int j = 0; j < 9; j++) {
							para2->data[i][j] = para1->data[i][j];
						}
					}
					para2->data[para1->n / 9][para1->n % 9] = i;
					pthread_create(&pid[i], NULL, mythread, (void *)para2);
				}
			}
			for (int i = 1; i <= 9; i++) {
				if (Check(para1->data, para1->n, i) == true) {
					pthread_join(pid[i], NULL);
				}
			}
		}
		else {
			for (int i = 1; i <= 9; i++) {
				if (Check(para1->data, para1->n, i) == true) {
					Parameter * para2 = new Parameter;
					para2->n = para1->n + 1;
					for (int i = 0; i < 9; i++) {
						for (int j = 0; j < 9; j++) {
							para2->data[i][j] = para1->data[i][j];
						}
					}
					para2->data[para1->n / 9][para1->n % 9] = i;
					int count = 0;
					DFS(para2->data, para2->n, count);
				}
			}
		}
	}
}

int DFS(int data[9][9], int n, int &count) 
{
	if (n == 81) /* 所有的都符合，则说明找到一种解法 */
	{	
		sum += 1;
		count += 1;
		cout << "方案" << count << "：" << endl;
		Output(data);
		return 0;
	}
	
	if (data[n / 9][n % 9] != 0) /* 当前位不为空时跳过 */
		DFS(data, n + 1, count);
	else 
	{
		for (int i = 1; i <= 9; i++) 
		{
			if (Check(data, n, i) == true) 
			{
				data[n / 9][n % 9] = i;
				DFS(data, n + 1, count);
				data[n / 9][n % 9] = 0;
			}
		}
	}
}