#include<iostream>
#include<time.h>
#include <stdio.h>
#include <stdlib.h> 
#include<sys/time.h>
#include<arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
#include <algorithm>
#include <omp.h>

using namespace std;

/* 定义一个函数指针类型，用于指向其他函数 */
typedef void (*fun_ptr)();

const int N = 64;
const int M = 37;
const int TIMES = 122;
const int NUM_THREADS = 8;

float matrix[N][N] = { 0 };
float A[N][N];

void generateUpperTriangleMatrix(float matrix[][N], int n)
{
	srand(time(nullptr));
	for (int i = 0; i < n; i++)
	{
		for (int j = i; j < n; j++)
		{
			matrix[i][j] = rand() % 100;
		}
	}
}

void operateRows(float matrix[][N], int n, int m, int times)
{
	srand(time(nullptr));
	for (int t = 1; t <= times; t++)
	{
		float* indices = new float[m];
		for (int i = 0; i < m; i++)
		{
			indices[i] = rand() % n;
		}
		for (int i = 0; i < m; i++)
		{
			int index1 = indices[i];
			for (int j = i; j < m; j++)
			{
				int index2 = indices[j];
				if (index1 == index2) continue; // 对角线上的元素只加一次
				for (int k = index2; k < n; k++)
				{
					matrix[index1][k] += matrix[index2][k];
				}
			}
		}
		delete[] indices;
	}
}

void m_reset() {
	generateUpperTriangleMatrix(matrix, N);
	operateRows(matrix, N, M, TIMES);
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
}

void printMatrix(float matrix[][N], int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

void normal_gaussian()
{
	for (int k = 1; k < N; k++)
	{
		float ele = A[k][k];
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / ele;
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

void omp_gaussian()
{
	#pragma omp parallel for
	for (int k = 1; k < N; k++)
	{
		float ele = A[k][k];

	#pragma omp parallel for
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / ele;
		}

		A[k][k] = 1.0;

	#pragma omp parallel for
		for (int i = k + 1; i < N; i++)
		{
	#pragma omp parallel for
			for (int j = k + 1; j < N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}



void open_mp_default() {
	int i, j, k, ele;
#pragma omp parallel num_threads(thread_count), private(i, j, k, ele)
	for (k = 0; k < n; k++) {
#pragma omp single
		{
			ele = A[k][k];
			for (j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
		}
#pragma omp for
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void open_mp_static() {
	int i, j, k, ele;
#pragma omp parallel num_threads(thread_count), private(i, j, k, ele)
	for (k = 0; k < n; k++) {
#pragma omp single
		{
			ele = A[k][k];
			for (j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
		}
#pragma omp for schedule(static, 128)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void open_mp_dynamic() {
	int i, j, k, ele;
#pragma omp parallel num_threads(thread_count), private(i, j, k, ele)
	for (k = 0; k < n; k++) {
#pragma omp single
		{
			ele = A[k][k];
			for (j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
		}
#pragma omp for schedule(dynamic, 128)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

void open_mp_guided() {
	int i, j, k, ele;
#pragma omp parallel num_threads(thread_count), private(i, j, k, ele)
	for (k = 0; k < n; k++) {
#pragma omp single
		{
			ele = A[k][k];
			for (j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
		}
#pragma omp for schedule(guided, 128)
		for (i = k + 1; i < n; i++) {
			for (j = k + 1; j < n; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}


void omp_simd()
{
	int i, j, k, ele;
#pragma omp parallel num_threads(thread_count), private(i, j, k, ele)

	for (int k = 1; k < N; k++)
	{
#pragma omp single
		float ele = A[k][k];

#pragma omp parallel for
		for (int j = k + 1; j < N; j++)
		{
#pragma omp simd
			A[k][j] = A[k][j] / ele;
		}

		A[k][k] = 1.0;

#pragma omp for
		for (int i = k + 1; i < N; i++)
		{
#pragma omp simd
			for (int j = k + 1; j < N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}



void calculate_time(fun_ptr f) {

	struct timeval starttime, endtime;
	double timeuse;

	gettimeofday(&starttime, NULL);

	f();

	gettimeofday(&endtime, NULL);
	timeuse = 1000000 * (endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
	timeuse /= 1000000;/*转换成秒输出*/
	printf("timeuse=%f", timeuse);
	cout << endl;
}

int main()
{
	m_reset();

	cout << "普通：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(normal_gaussian);

	cout << "omp：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(omp_gaussian);

	return 0;
}

