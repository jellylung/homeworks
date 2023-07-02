#include<iostream>
#include<time.h>
#include <stdio.h>
#include <stdlib.h> 
#include<sys/time.h>
#include<arm_neon.h>
#include <pthread.h>
#include <semaphore.h>
#include <algorithm>

using namespace std;

/* 定义一个函数指针类型，用于指向其他函数 */
typedef void (*fun_ptr)();

const int N = 64;
const int M = 37;
const int TIMES = 122;
const int NUM_THREADS = 8;

float matrix[N][N] = { 0 };
float A[N][N];

pthread_mutex_t mutex[N];


typedef struct {
	int k; //消去的轮次
	int t_id; // 线程 id
}threadParam_t;

typedef struct {

	float A[N][N];
	int start_row;  // 起始行
	int end_row;    // 结束行（不包括）
	int k;
} ThreadData;

sem_t sem_leader;
sem_t sem_Divsion[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

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

void* threadFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	for (int j = k + 1; j < N; ++j) {
		A[i][j] = A[i][j] - A[i][k] * A[k][j];
	}
	A[i][k] = 0;
	pthread_exit(NULL);
}

void gaussian_pth()
{
	for (int k = 0; k < N; ++k)
	{
		float ele = matrix[k][k];
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / ele;
		}
		A[k][k] = 1.0;

		int worker_count = 8; //工作线程数量
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}

		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_create(handles + t_id, NULL, &threadFunc, param + t_id);
		}

		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			pthread_join(handles[t_id], NULL);
		}
	}
}

void* threadFunc_ss(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k) {
		if (t_id == 0) {
			for (int j = k + 1; j < N; j++) {
				A[k][j] = A[k][j] / A[k][k];
			}
			A[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		if (t_id == 0) {// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
			for (int i = 0; i < NUM_THREADS - 1; ++i) {
				sem_post(&sem_Divsion[i]);
			}
		}

		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {//循环划分任务

			for (int j = k + 1; j < N; ++j) {//消去
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
		}
		if (t_id == 0) {
			for (int i = 0; i < NUM_THREADS - 1; ++i) {
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}
			for (int i = 0; i < NUM_THREADS - 1; ++i) {
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else {
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
}

void pth_ss()
{
	sem_init(&sem_leader, 0, 0);//初始化信号量

	for (int i = 0; i < NUM_THREADS; ++i) {
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}//创建线程

	pthread_t handles[NUM_THREADS];// 创建对应的 Handle
	threadParam_t param[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(handles + t_id, NULL, &threadFunc_ss, param + t_id);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handles[t_id], NULL);
	}

	sem_destroy(&sem_leader);	//销毁所有信号量
}

void* threadFunc_ba(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k) {// t_id 为 0 的线程做除法操作，其它工作线程先等待

		if (t_id == 0) {// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
			for (int j = k + 1; j < N; j++) {
				A[k][j] = A[k][j] / A[k][k];

			}
			A[k][k] = 1.0;
		}

		pthread_barrier_wait(&barrier_Divsion);//第一个同步点

		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {//循环划分任务(其他方法0

			for (int j = k + 1; j < N; ++j) {//消去
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
		}

		pthread_barrier_wait(&barrier_Elimination);// 第二个同步点
	}
	pthread_exit(NULL);
}

void pth_ba()
{
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);//初始化 barrier
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

	pthread_t handles[NUM_THREADS];//创建线程// 创建对应的 Handle
	threadParam_t param[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(handles + t_id, NULL, &threadFunc_ba, param + t_id);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handles[t_id], NULL);
	}
	pthread_barrier_destroy(&barrier_Elimination);
	pthread_barrier_destroy(&barrier_Divsion);//销毁所有的 barrier	
}

void* thread_worker(void* arg) {
	int thread_id = *(int*)arg;
	int start = thread_id * (N - 1) / 8 + 1;
	int end = (thread_id + 1) * (N - 1) / 8 + 1;

	for (int k = start; k < end; k++) {
		pthread_mutex_lock(&mutex[k]);
		float ele_k = matrix[k][k];
		for (int j = k + 1; j < N; j++) {
			matrix[k][j] = matrix[k][j] / ele_k;
		}
		matrix[k][k] = 1.0;
		pthread_mutex_unlock(&mutex[k]);

		for (int i = k + 1; i < N; i++) {
			pthread_mutex_lock(&mutex[i]);
			for (int j = k + 1; j < N; j++) {
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
			}
			pthread_mutex_unlock(&mutex[i]);

			if (thread_id == 0) {
				matrix[i][k] = 0;
			}
		}
	}
	return NULL;
}

void pth_mutex()
{
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];

	/*初始化互斥锁*/
	for (int i = 0; i < N; i++) {
		pthread_mutex_init(&mutex[i], NULL);
	}

	/*创建线程*/
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, thread_worker, (void*)&thread_ids[i]);
	}

	/* 等待线程结束*/
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	/* 销毁互斥锁*/
	for (int i = 0; i < N; i++) {
		pthread_mutex_destroy(&mutex[i]);
	}
}

void* pthread_func_rr(void* arg)
{
	ThreadData* td = (ThreadData*)arg;
	int start_row = td->start_row;
	int end_row = td->end_row;
	int k = td->k;

	for (int i = start_row; i < end_row; i++)
	{
		for (int j = k + 1; j < N; j++)
		{
			A[i][j] = A[i][j] - A[i][k] * A[k][j];
		}
		A[i][k] = 0;
	}

	return NULL;
}

void pth_rr()
{
	pthread_t threads[NUM_THREADS];
	ThreadData td[NUM_THREADS];

	for (int k = 0; k < N; k++)
	{
		float ele = A[k][k];
		for (int j = k + 1; j < N; j++)
		{
			A[k][j] = A[k][j] / ele;
		}
		A[k][k] = 1.0;

		/* 拆分任务并启动线程*/
		int chunk_size = (N - k - 1) / NUM_THREADS + 1;
		for (int i = 0; i < NUM_THREADS; i++)
		{
			td[i].start_row = k + 1 + i * chunk_size;
			td[i].end_row = k + 1 + (i + 1) * chunk_size;
			td[i].k = k;

			if (td[i].end_row > N)
				td[i].end_row = N;

			pthread_create(&threads[i], NULL, pthread_func_rr, &td[i]);
		}

		/* 等待所有线程执行完成*/
		for (int i = 0; i < NUM_THREADS; i++)
		{
			pthread_join(threads[i], NULL);
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

	cout << "pthread动态：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(gaussian_pth);

	cout << "pthread静态：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(pth_ss);

	cout << "barrier：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(pth_ba);

	cout << "mutex：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(pth_mutex);

	cout << "拆分：";
	copy(&matrix[0][0], &matrix[0][0] + N * N, &A[0][0]);
	calculate_time(pth_rr);

	return 0;
}

