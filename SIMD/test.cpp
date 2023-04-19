#include<iostream>
#include<time.h>
#include <stdio.h>
#include <stdlib.h> 
#include<sys/time.h>
#include<arm_neon.h>

using namespace std;
const int N=2000;
float m[N][N];
float ma[N][N];
float mb[N][N];
float mc[N][N];
float md[N][N];
void m_reset() {
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m[i][j] = 0.0;
	    ma[i][j] = 0.0; mb[i][j] = 0.0; mc[i][j] = 0.0; md[i][j] = 0.0;
        }
        m[i][i] = 1.0;
	ma[i][i] = 1.0; 
	mb[i][i] = 1.0; 
	mc[i][i] = 1.0;  
	md[i][i] = 1.0;
        for (int j = i + 1; j < N; j++){
            m[i][j] = rand();
	    ma[i][j] =m[i][j];
        mb[i][j] =m[i][j];
	mc[i][j] =m[i][j];
	md[i][j] =m[i][j];
	}
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
            {    m[i][j] += m[k][j];
	ma[i][j] =m[i][j]; mb[i][j] =m[i][j]; mc[i][j] =m[i][j]; md[i][j] =m[i][j];
}
}

void normal_gaussian(float matrix[][N]) 
{
    for (int k = 1; k < N; k++)
    {
        for (int j = k+1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}



void neon_4to6(float matrix[][N])
{
    float32x4_t vec1;
    float32x4_t   s1, s2, pro;
    float32x4_t tem, divisor;
    for (int k = 1; k < N; k++)
    {
        divisor = vdupq_n_f32(matrix[k][k]);
        for (int j = k+1, j_end = N - N % 8; j < j_end; j += 4) {
            vec1 = vld1q_f32(matrix[k]+j); 
            vec1 = vdivq_f32(vec1,divisor); 
            vst1q_f32(matrix[k]+j, vec1); 
        }
       for (int j = (N - N%3); j < N; j++) {
                 matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
     matrix[k][k] = 1.0;
     for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}


void neon_8to13(float matrix[][N])
{
    float32x4_t   s1, s2, pro;
    float32x4_t tem, divisor;
    for (int k = 1; k < N; k++)
    {
        for (int j = k+1; j < N; j++)
        {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) 
        {
            s1 = vdupq_n_f32(matrix[i][k]); 
            for (int j = k+1, j_end = N - N % 8; j < j_end; j += 4) {
                s2 = vld1q_f32(matrix[k]+j); 
                tem = vld1q_f32(matrix[i]+j);
                pro = vmulq_f32(s1, s2); 
	tem= vsubq_f32(tem,pro);
                vst1q_f32(matrix[i]+j, tem); 
            }
            for (int j = (N - N%3); j < N; j++) {
                matrix[i][j] -= matrix[k][j] * matrix[i][k];
            }
            matrix[i][k] = 0.0f;
        }
    }
}

void neon_gaussian(float matrix[][N])
{
    float32x4_t vec1, vec2; 
    float32x4_t   s1, s2, pro;
    float32x4_t tem, divisor;
    for (int k = 1; k < N; k++)
    {
        divisor = vdupq_n_f32(matrix[k][k]);
        for (int j = k+1, j_end = N - N % 8; j < j_end; j += 4) {
            vec1 = vld1q_f32(matrix[k]+j); 
            vec1 = vdivq_f32(vec1,divisor); 
            vst1q_f32(matrix[k]+j, vec1); 
        }
       for (int j = (N - N%3); j < N; j++) {
                 matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            s1 = vdupq_n_f32(matrix[i][k]); 
            for (int j = k+1, j_end = N - N % 8; j < j_end; j += 4) {
                s2 = vld1q_f32(matrix[k]+j); 
                tem = vld1q_f32(matrix[i]+j);
                pro = vmulq_f32(s1, s2); 
	tem= vsubq_f32(tem,pro);
                vst1q_f32(matrix[i]+j, tem); 
            }
            for (int j = (N - N%3); j < N; j++) {
                matrix[i][j] -= matrix[k][j] * matrix[i][k];
            }
            matrix[i][k] = 0.0f;
        }
    }
}


int main()
{
    m_reset();

    cout << endl << "普通串行的高斯消去法：" << endl;
    struct timeval starttime,endtime;
        double timeuse;
        gettimeofday(&starttime,NULL);

         normal_gaussian(ma);

        gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-   starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout << "总共耗时： ";
        printf("timeuse=%f",timeuse);
    cout << endl;



    cout << endl << "neon的高斯消去法：" << endl;

    gettimeofday(&starttime,NULL);

    neon_gaussian(mb);

        gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-   starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout << "总共耗时： ";
        printf("timeuse=%f",timeuse);
    cout << endl;


    cout << endl << "4到6行的向量化：" << endl;
    gettimeofday(&starttime,NULL);

    neon_4to6(mc);

        gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-   starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout << "总共耗时： ";
        printf("timeuse=%f",timeuse);
    cout << endl;


    cout << endl << "8到13行的向量化：" << endl;
    gettimeofday(&starttime,NULL);

    neon_8to13(md);

        gettimeofday(&endtime,NULL);
        timeuse=1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-   starttime.tv_usec;
        timeuse/=1000000;/*转换成秒输出*/
        cout << "总共耗时： ";
        printf("timeuse=%f",timeuse);
    cout << endl;

    return 0;

}














