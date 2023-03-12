#include <iostream>
#include<Windows.h>
#include<stdlib.h>
using namespace std;

const int N = 10240;
double a[N], b[N][N], sum[N];

void init(int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i][j] = i + j;
            a[i] = i + j;
        }
    }
}

int main()
{
    long long head, tail, freq;
    int n;
    int times;
    cin>>n>>times;
    init(n);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    int i,j;
    for(int k=0;k<times;k++){
        for (i = 0; i < n; i++)
        {
            sum[i] = 0.0;
        }
        for (j = 0; j< n; j++)
        {
            for (i = 0; i < n; i++) {
                sum[i] += b[j][i] * a[j];
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout <<"Col:"<< ((tail-head) *1000.0/freq)
        << "ms" << endl;
        cout<<sum[3];
    return 0;
}
