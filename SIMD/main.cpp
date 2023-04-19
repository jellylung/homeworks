#include<iostream>
#include<time.h>
#include <stdio.h>
#include <immintrin.h> // 包含 AVX2 指令集
#include <windows.h>
#include <stdlib.h>

//#include<arm_neon.h>

using namespace std;

const int N = 1024;
float m[N][N];

void m_reset() {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m[i][j] = 0.0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}

void normal_gaussian(float A[][N])
{
    for (int k = 1; k < N; k++)
    {

        for (int j = k+1; j < N; j++)
        {
            //if (m[k][k] == 0)
            //    continue;
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k+1; i < N; i++)
        {
            for (int j = k+1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void avx_gaussian(float A[][N])
{
    __m256 t1, t2, res;
    __m256 s1, s2, pro;
    __m256 tem, diff;
    for (int k = 1; k < N; k++)
    {
        t1 = _mm256_loadu_ps(&A[k][k]); // 用来保存矩阵 A[k][k] 的向量
        for (int j = k+1; j < N; j += 8) {
            t2 = _mm256_loadu_ps(A[k] + j); // 从内存中读取量级矩阵 A[k][j] 的向量
            res = _mm256_div_ps(t2, t1); // 计算向量相除结果
            _mm256_storeu_ps(A[k] + j, res); // 将结果写回内存中
        }

        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            s1 = _mm256_loadu_ps(A[i] + k); // 用来保存矩阵 A[i][k] 的向量
            for (int j = k + 1, j_end = N - N % 8; j < j_end; j += 8) {
                tem = _mm256_loadu_ps(A[i] + j); // 从内存中读取量级矩阵 A[i][j] 的向量
                s2 = _mm256_loadu_ps(A[k]+j); // 从内存中读取量级矩阵 A[k][j] 的向量
                pro = _mm256_mul_ps(s1, s2);

                diff = _mm256_sub_ps(tem, pro);
                _mm256_storeu_ps(A[i] + j, diff); // 将结果写回内存中
            }
            for (int j = N - N % 8; j < N; j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0;
        }
    }
}

int main() {
    m_reset();
    //float m[4][5] = { {0}, {0,1,1,1,3},{0,1,2,4,7},{0,1,3,9,13} };

    //for (int i = 1; i < N; i++)
    //{
    //    for (int j = 1; j < N; j++)
    //    {
    //        cout << m[i][j] << " ";
    //    }
    //    cout << endl;
    //}

    cout << endl;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) & freq);// start time
    QueryPerformanceCounter((LARGE_INTEGER *) & head);
    normal_gaussian(m);
    QueryPerformanceCounter((LARGE_INTEGER *) & tail);
    cout << "总共耗时： ";
    cout << (tail -head) * 1000.0 / freq
         << "ms" << endl;

    //for (int i = 1; i < N; i++)
    //{
    //    for (int j = 1; j < N; j++)
    //    {
    //        cout << m[i][j] << " ";
    //    }
    //    cout << endl;
    //}

    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);// start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    avx_gaussian(m);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "总共耗时： ";
    cout << (tail - head) * 1000.0 / freq
        << "ms" << endl;

    cout << endl;
    //for (int i = 1; i < N; i++)
    //{
    //    for (int j = 1; j < N; j++)
    //    {
    //        cout << m[i][j] << " ";
    //    }
    //    cout << endl;
    //}
    return 0;
}
