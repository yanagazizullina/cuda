#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>

using namespace std;
#define BLOCK_SIZE 16
const unsigned int N = 2000;
float a[N][N];
float b[N][N];
float arrayGPU[N][N];
float arrayCPU[N][N];


__global__ void multipleGPU(float* a, float* b, int n, float* c)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float sum = 0.0f;
	int ia = n * BLOCK_SIZE * by + n * ty;
	int ib = BLOCK_SIZE * bx + tx;
	int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	for (int k = 0; k < n; k++)
		sum += a[ia + k] * b[ib + k * n];
	c[ic + n * ty + tx] = sum;
}

int main()
{
	float timerGPU;
	//float timerCPU;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = rand() % 10 * sizeof(float);
			b[i][j] = rand() % 10 * sizeof(float);
		}
	}
	/*
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(4) << a[i][j];
		}
		cout << endl;
	}

	printf("-----------------------------\n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(4) << b[i][j];
		}
		cout << endl;
	}*/


	//Matrix multiplication by CPU
	//cudaEventRecord(start, 0);
	clock_t start_s = clock();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			arrayCPU[i][j] = 0;
			for (int k = 0; k < N; k++) {
				arrayCPU[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	clock_t stop_s = clock();
	cout << "\n CPU time " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " msec" << endl;
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&timerCPU, start, stop);
	/*printf("--------------Multiple CPU---------------\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(4) << arrayCPU[i][j];
		}
		cout << endl;
	}*/


	//Matrix multiplication by GPU
	int size = N * N * sizeof(float);
	float* da, * db, * dc;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);
	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	multipleGPU << <blocks, threads >> > (da, db, N, dc);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerGPU, start, stop);

	cudaThreadSynchronize();
	cudaMemcpy(arrayGPU, dc, size, cudaMemcpyDeviceToHost);
	/*printf("--------------Multiple GPU---------------\n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(4) << arrayGPU[i][j];
		}
		cout << endl;
	}*/
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	//printf("\n CPU time %f msec\n", timerCPU);
	printf("\n GPU time %f msec\n", timerGPU);
}
