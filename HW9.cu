// Name: Robert Barrett
// Vector Dot product on many blocks using shared memory
// nvcc HW9.cu -o temp

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 1000 // Change the length of the vector - try different values

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU arrays
float *A_GPU, *B_GPU, *C_GPU; //GPU arrays
float DotCPU, DotGPU;
dim3 BlockSize; // Dimensions of your blocks
dim3 GridSize; // Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *file, int line);
void setUpDevices();
void allocateMemory(int numBlocks);
void innitialize();
void dotProductCPU(float*, float*, float*, int);
__global__ void dotProductGPU(const float*, const float*, float*, int);
bool check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp(int numBlocks);

// This check to see if an error happened in your CUDA code
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 200;  // Fixed thread count as requested
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;  // Will be updated in main based on N
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory(int numBlocks)
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	// C_CPU will store partial sums from each block
	C_CPU = (float*)malloc(numBlocks * sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU, numBlocks * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// CPU version - computes full dot product
void dotProductCPU(float *a, float *b, float *c_out, int n)
{
	float acc = 0.0f;
	for(int i = 0; i < n; i++)
	{ 
		acc += a[i] * b[i];
	}
	c_out[0] = acc;
}

/*
GPU kernel: Each block computes partial dot product using shared memory reduction
Each thread computes one element product, then we reduce within the block
Thread 0 of each block writes the partial sum to c[blockIdx.x]
*/
__global__ void dotProductGPU(const float *a, const float *b, float *c, int n)
{
	// Dynamic shared memory for this block
	extern __shared__ float sharedData[];
	
	int tid = threadIdx.x;  // Thread ID within block
	int globalIdx = blockIdx.x * blockDim.x + tid;  // Global thread index
	
	// Load data into shared memory (0 if out of bounds)
	float val = 0.0f;
	if(globalIdx < n) {
		val = a[globalIdx] * b[globalIdx];
	}
	sharedData[tid] = val;
	__syncthreads();
	
	// Sequential reduction - simple and works for any block size
	// Only thread 0 does the reduction to avoid race conditions
	if(tid == 0) {
		float sum = 0.0f;
		for(int i = 0; i < blockDim.x; i++) {
			sum += sharedData[i];
		}
		c[blockIdx.x] = sum;
	}
}

// Checking to see if results match within tolerance
bool check(float cpuAnswer, float gpuAnswer, float tolerance)
{
	if(cpuAnswer == 0.0f) {
		printf("\n\n absolute error = %f\n", fabs(gpuAnswer - cpuAnswer));
		return fabs(gpuAnswer - cpuAnswer) < tolerance;
	}
	
	double percentError = fabs((gpuAnswer - cpuAnswer) / cpuAnswer) * 100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	return percentError < tolerance;
}

// Calculating elapsed time
long elaspedTime(struct timeval start, struct timeval end)
{
	long startTime = start.tv_sec * 1000000 + start.tv_usec;
	long endTime = end.tv_sec * 1000000 + end.tv_usec;
	return endTime - startTime;
}

// Cleaning up memory
void CleanUp(int numBlocks)
{
	if (A_CPU) free(A_CPU);
	if (B_CPU) free(B_CPU);
	if (C_CPU) free(C_CPU);

	if (A_GPU) cudaFree(A_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	if (B_GPU) cudaFree(B_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	if (C_GPU) cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	printf("Vector size N = %d\n", N);
	
	// Setting up the GPU
	setUpDevices();
	
	// Compute number of blocks needed
	int threadsPerBlock = BlockSize.x;
	int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
	GridSize.x = numBlocks;
	
	printf("Using %d blocks with %d threads per block\n", numBlocks, threadsPerBlock);
	printf("Total threads: %d\n", numBlocks * threadsPerBlock);
	
	// Allocate memory
	allocateMemory(numBlocks);
	
	// Initialize vectors
	innitialize();
	
	// Compute on CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Verify the expected mathematical result
	double expected = 0.0;
	for(int i = 0; i < N; i++) {
		expected += (double)i * (double)i * 3.0;
	}
	printf("Expected result (mathematical): %.1f\n", expected);
	
	// Compute on GPU
	gettimeofday(&start, NULL);
	
	// Copy data to GPU
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Launch kernel with shared memory
	size_t sharedBytes = threadsPerBlock * sizeof(float);
	dotProductGPU<<<numBlocks, threadsPerBlock, sharedBytes>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy partial results back to CPU
	cudaMemcpy(C_CPU, C_GPU, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Final reduction on CPU - sum all partial results
	DotGPU = 0.0f;
	printf("\nPartial sums from each block:\n");
	for(int i = 0; i < numBlocks; i++) {
		printf("Block %d: %f\n", i, C_CPU[i]);
		DotGPU += C_CPU[i];
	}
	printf("Total blocks: %d\n", numBlocks);
	
	// Manual verification of first block to debug
	float manualBlock0 = 0.0f;
	for(int i = 0; i < 200 && i < N; i++) {
		manualBlock0 += A_CPU[i] * B_CPU[i];
	}
	printf("Manual calculation for block 0: %f\n", manualBlock0);
	
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Check results
	printf("\nCPU Result: %f", DotCPU);
	printf("\nGPU Result: %f", DotGPU);
	
	if(check(DotCPU, DotGPU, Tolerance)) {
		printf("\n\nDot product computed correctly on GPU!");
		printf("\nCPU time: %ld microseconds", timeCPU);
		printf("\nGPU time: %ld microseconds", timeGPU);
		if(timeCPU > 0) {
			printf("\nSpeedup: %.2fx", (float)timeCPU / timeGPU);
		}
	} else {
		printf("\n\nERROR: GPU and CPU results don't match!");
	}
	
	// Cleanup
	CleanUp(numBlocks);	
	
	printf("\n\n");
	return 0;
}
