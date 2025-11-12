// Name:
// Vector addition on two GPUs.
// nvcc HW22a.cu -o temp
/*
 Purpose:
 To learn how to use multiple GPUs.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU0, *B_GPU0, *C_GPU0; //GPU0 pointers
float *A_GPU1, *B_GPU1, *C_GPU1; // GPU1 pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
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
	//Check GPU availability; if at least two aren't avaliable, quit. Otherwise pick 
  // GPU 0 and 1. 
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(deviceCount < 2)
	{
		printf("\n ERROR: Need at least 2 GPUs, but only %d available\n", deviceCount);
		exit(1);
	}
	
	printf("\n Found %d GPUs. Using GPU 0 and GPU 1.\n", deviceCount);
  
  //Calculate half sizes (first half gets extra element if odd). Doing it this way makes sure N_half1 + N_half2 = N
	int N_half1 = (N + 1) / 2;// First half (gets extra element if N is odd)
	int N_half2 = N / 2;// Second half
  
  BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Calculate grid sizes for each GPU based on their portion
	GridSize0.x = (N_half1 - 1)/BlockSize.x + 1;
	GridSize0.y = 1;
	GridSize0.z = 1;
	
	GridSize1.x = (N_half2 - 1)/BlockSize.x + 1;
	GridSize1.y = 1;
	GridSize1.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
  //Calculate half sizes
	int N_half1 = (N + 1) / 2;
	int N_half2 = N / 2;
  
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	//Allocate memory on GPU 0 for first half
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A_GPU0, N_half1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU0, N_half1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU0, N_half1*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

  //Allocate memory on GPU 1 for second half
	cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A_GPU1, N_half2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU1, N_half2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU1, N_half2*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// Because of both the way we are copying the memory later in main and splitting up the task between 2 GPUs, 
// we don't need to change the kernel. Each GPU doesn't know which 'part' of the vector its working on, 
// they get an adjacent, non-overlapping chunk of data and processes it. 
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);

	//Free GPU 0 memory
	cudaSetDevice(0);
	cudaFree(A_GPU0); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU0); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Free GPU 1 memory
	cudaSetDevice(1);
	cudaFree(A_GPU1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU1); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU1);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;

  //Calculate half sizes
	int N_half1 = (N + 1) / 2;// First half
	int N_half2 = N / 2; // Second half
  
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
  
	//Copy first half to GPU 0. GPU 0 gets the first 'half' of the vector's elements
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(A_GPU0, A_CPU, N_half1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU0, B_CPU, N_half1*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

  // Launch kernel on GPU 0
	addVectorsGPU<<<GridSize0,BlockSize>>>(A_GPU0, B_GPU0, C_GPU0, N_half1);
	cudaErrorCheck(__FILE__, __LINE__);

  //Copy second half to GPU 1. GPU 1 gets the second 'half' of the vector's elements. 
	cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(A_GPU1, A_CPU + N_half1, N_half2*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU1, B_CPU + N_half1, N_half2*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Launch kernel on GPU 1
	addVectorsGPU<<<GridSize1,BlockSize>>>(A_GPU1, B_GPU1, C_GPU1, N_half2);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copy results from GPU 0
	cudaSetDevice(0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(C_CPU, C_GPU0, N_half1*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copy results from GPU 1
	cudaSetDevice(1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(C_CPU + N_half1, C_GPU1, N_half2*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Synchronize both GPUs
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

  
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
