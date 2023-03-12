#include <iostream>
#include<Windows.h>
#include<stdlib.h>

using namespace std;
const int N = 100240;
double a[N];

void init(int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i ;
    }
}

int main()
{
    long long head, tail, freq;

    int n,sum=0;
    int times;
    cin>>n>>times;
    init(n);
    int sum1=0,sum2=0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int j=0;j<times;j++){
        for (int i = 0; i < n; i+=2)
        {
            sum1 += a[i];
            sum2 += a[i+1];
        }
        sum=sum1+sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout <<"Col:"<< ((tail-head) *1000.0/freq)
        << "ms" << endl;
        cout<<sum;
    return 0;
}
