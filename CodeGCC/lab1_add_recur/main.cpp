#include <iostream>
#include<Windows.h>
#include<stdlib.h>

using namespace std;
const int N = 100240;
double a[N] = { 0 };

void init(int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
}

int main()
{
    long long head, tail, freq;

    int n;
    int times;
    int m, i,j;
    cin >> n >> times;
    init(n);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(j=0;j<times;j++){
        for (m = n; m > 0; m /= 2) {
            for (i = 0; i < m ; i++) {
                a[i] = a[i * 2] + a[i * 2 + 1];
            }
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "Col:" << ((tail - head) * 1000.0 / freq)
        << "ms" << endl;
    cout << a[0];
    return 0;
}
