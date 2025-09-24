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

void setUpDevices()
{
	BlockSize.x = 200;  
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;  // Will be updated in main based on N
	GridSize.y = 1;
	GridSize.z = 1;
}

/*Books storage on both the host and device
On the host: A_CPU and B_CPU hold the input vectors, while C_CPU holds the partial sums from each block
On the device: A_GPU and B_GPU hold the input vectors, while C_GPU holds the partial sums from each block*/
void allocateMemory(int numBlocks)
{	
	// Host memory			
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	// C_CPU will store partial sums from each block
	C_CPU = (float*)malloc(numBlocks * sizeof(float));
	
	// Device Memory
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

// Iterates through the vectors and computes the dot product on the CPU, storing the result in c_out[0]
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
(1) Each block gets some shared memory which all its threads can access quickly 
(2) Each thread multiplies its corresponding elements from a and b, storing the result in shared memory
(3) Waits until each thread is finished
(4) One thread (thread 0) sums up all the values in shared memory and writes the result to c[blockIdx.x]
*/
__global__ void dotProductGPU(const float *a, const float *b, float *c, int n)
{
	
    /*
    Dynamic memory is used here so we don't have to change the array manually if the BlockSize changes;
    rather, the shared memory array's size is not fixed at compile time, but rather it's determined 
    when the kernel launches. 
    */
   // (1)
	extern __shared__ float sharedData[];

	//(2)
    // Thread ID within block
	int tid = threadIdx.x;  
    // Global thread index
	int globalIdx = blockIdx.x * blockDim.x + tid;  
	
	// Load data into shared memory (0 if out of bounds)
	float val = 0.0f;
	if(globalIdx < n) {
		val = a[globalIdx] * b[globalIdx];
	}
	sharedData[tid] = val;
    //(3)
	__syncthreads();
	
    // (4)
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
/*
(1) Sets up the grid by computing how many blocks are needed 
(2) Allocate memory and initialize the vectors w/ data
(3) Perform the dot product on the CPU 
(4) Check the math was consistent 
(5) Perform the dot product on the GPU 
    (a) Copy inputs from the CPU on to the GPU 
    (b) Launch the kernel with numBlocks blocks and BlockSize threads per block, using shared memory
    (c) Copy partial results sums back 
    (d) Add the partial sums on the CPU 
(6) Print results 
    (a) Show partial results from each block 
    (b) Show the final result from the GPU and CPU
    (c) Show the time taken by each and the speedup
(7) Cleanup memory 
*/
int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	printf("Vector size N = %d\n", N);
	
	setUpDevices();
	
	//(1)
	int threadsPerBlock = BlockSize.x;
	int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
	GridSize.x = numBlocks;
	
	printf("Using %d blocks with %d threads per block\n", numBlocks, threadsPerBlock);
	printf("Total threads: %d\n", numBlocks * threadsPerBlock);
	
	//(2)
	allocateMemory(numBlocks);
	
	// Initialize vectors
	innitialize();
	
	// (3)
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// (4)
	double expected = 0.0;
	for(int i = 0; i < N; i++) {
		expected += (double)i * (double)i * 3.0;
	}
	printf("Expected result (mathematical): %.1f\n", expected);
	
	// (5)
	gettimeofday(&start, NULL);
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	/*
    size_t is an unsigned integer type which represents the size of any object in bytes, which is 
    why it's appropriate to use here because we're asking 'how many bytes of shared memory do we need?'
    Here, threadsPerBlock = 200, and sizeof(float) = 4 bytes, so sharedBytes = 800 bytes. 
    This is used in dotProductGPU as the third argument which tells CUDA when launching the kernel
    how much dynamic shared memory to allocate for each block.
    This memory is used in sharedData[] inside each block. 
    */
	size_t sharedBytes = threadsPerBlock * sizeof(float);
	dotProductGPU<<<numBlocks, threadsPerBlock, sharedBytes>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
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
	
	// (6)
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
	
	// (7)
	CleanUp(numBlocks);	
	
	printf("\n\n");
	return 0;
}
