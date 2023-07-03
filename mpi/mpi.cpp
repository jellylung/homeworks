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


void block_run(int version) {
	//块划分
	void (*f)(int, int);
	string inform = "";
	if (version == 0) {
		f = &block_gauss;
		inform = "block assign time is: ";
	}
	else if (version == 1) {
		f = &block_gauss_opt;
		inform = "block assign opt time is: ";
	}
	timeval begin, finish;

	int num_proc;
	int my_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int block_size = n / num_proc;
	int remain = n % num_proc;
	if (my_rank == 0) {
		arr_reset();
		gettimeofday(&begin, NULL);
		for (int i = 1; i < num_proc; i++) {
			int upper_bound = i != num_proc - 1 ? block_size : block_size + remain;
			for (int j = 0; j < upper_bound; j++)
				MPI_Send(A[i * block_size + j], n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		}
		f(my_rank, num_proc);
		for (int i = 1; i < num_proc; i++) {
			int upper_bound = i != num_proc - 1 ? block_size : block_size + remain;
			for (int j = 0; j < upper_bound; j++)
				MPI_Recv(A[i * block_size + j], n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		testResult();
		gettimeofday(&finish, NULL);
		cout << inform << millitime(finish) - millitime(begin) << "ms" << endl;
	}
	else {
		int upper_bound = my_rank != num_proc - 1 ? block_size : block_size + remain;
		for (int j = 0; j < upper_bound; j++)
			MPI_Recv(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		f(my_rank, num_proc);
		for (int j = 0; j < upper_bound; j++)
			MPI_Send(A[my_rank * block_size + j], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
	}
}

void block_gauss(int my_rank, int num_proc) {
	int block_size = n / num_proc;
	int remain = n % num_proc;

	int my_begin = my_rank * block_size;
	int my_end = my_rank == num_proc - 1 ? my_begin + block_size + remain : my_begin + block_size;
	for (int k = 0; k < n; k++) {
		if (k >= my_begin && k < my_end) {
			float ele = A[k][k];
			for (int j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
			for (int p = my_rank + 1; p < num_proc; p++)
				MPI_Send(A[k], n, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
		}
		else {
			int current_work_p = k / block_size;
			if (current_work_p < my_rank)
				MPI_Recv(A[k], n, MPI_FLOAT, current_work_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		for (int i = my_begin; i < my_end; i++) {
			if (i > k) {
				for (int j = k + 1; j < n; j++) {
					A[i][j] = A[i][j] - A[i][k] * A[k][j];
				}
				A[i][k] = 0.0;
			}
		}
	}

}


void pip_gauss(int my_rank, int num_proc) {
	int pre_rank = (my_rank - 1 + num_proc) % num_proc;
	int nex_rank = (my_rank + 1) % num_proc;
	for (int k = 0; k < n; k++) {
		if (k % num_proc == my_rank) {
			float ele = A[k][k];
			for (int j = k + 1; j < n; j++)
				A[k][j] = A[k][j] / ele;
			A[k][k] = 1.0;
			if (nex_rank != my_rank)
				MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
		}
		else {
			MPI_Recv(A[k], n, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (k % num_proc != nex_rank)
				MPI_Send(A[k], n, MPI_FLOAT, nex_rank, 2, MPI_COMM_WORLD);
		}
		for (int i = my_rank; i < n; i += num_proc) {
			if (i > k) {
				for (int j = k + 1; j < n; j++) {
					A[i][j] = A[i][j] - A[i][k] * A[k][j];
				}
				A[i][k] = 0.0;
			}
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

	cout << "block：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(block_gauss);

	cout << "pip：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(pip_gauss);

	return 0;
}

