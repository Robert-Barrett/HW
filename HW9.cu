// Name: Robert Barrett
// Vector Dot product on many block and using shared memory
// nvcc HW9.cu -o temp


// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 25 // Change the length of the vector

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
 // Amount of memory now depends on how many block needed
void allocateMemory(int numBlocks);
void innitialize();
void dotProductCPU(float*, float*,float*, int);
__global__ void dotProductGPU(const float*,const float*,const float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
//Amount of memory cleaned up depends on the number of blocks
void CleanUp(int numBlocks);

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
	BlockSize.x = 200;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory(int numBlocks)
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
    //// C_CPU will be used to receive partial sums from device (numBlocks floats)
	C_CPU = (float*)malloc(numBlocks * sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,numBlocks * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}


// CPU version (computes full dot product and stores partials like original)
void dotProductCPU(float *a, float *b, float *C_cpu_out, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_cpu_out[id] = a[id] * b[id];
	}
	//reduce into C_cpu_out
    float acc = 0.0f;
	for(int id = 0; id < n; id++)
	{ 
		acc += C_cpu_out[id];
	}
    C_cpu_out[0] = acc;
}

/*
Each thread computes the product for global index idx (if idx < n)
Stores the results into shared memory then does reduction within the blocks
thread 0 writes the block's partial sum to c[blockIdx.x]
*/
__global__ void dotProductGPU(const float *a, const float *b, float *c, int n)
{
	// Dynamic Shared Memory
    extern __shared__ float sharedData[];
    int id = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + id;
	

    // load product into shared memory (0 if the memory is out of range)
	float val = 0.0f;
    if(globalIdx < n)
        val = a[globalIdx] * b[globalIdx];
    sharedData[id] = val;
    __syncthreads();
   /*
   In-block reduction 
   Reduces blockDim.x 
   */
	int fold = blockDim.x / 2;
	while(1 < fold)
	{
		if(fold%2 != 0)
		{
			if(id == 0 && (fold - 1) < n)
			{
				c[0] = c[0] + c[fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(id < fold && (id + fold) < n)
		{
			c[id] = c[id] + c[id + fold];
		}
		__syncthreads();
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
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
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
    ///compute number of blocks needed
    int threadsPerBlock = BlockSize.x;
    int numBlocks = (N + threadsPerBlock -1) / threadsPerBlock;
    GridSize.x = numBlocks; 
	
	// Allocating the memory you will need.
	allocateMemory(numBlocks);
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	/*if(BlockSize.x < N)
	{
		printf("\n\n Your vector size is larger than the block size.");
		printf("\n Because we are only using one block this will not work.");
		printf("\n Good Bye.\n\n");
		exit(0);
	}
	*/
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
    //launch kernel with dynamic shared mem = threadsPerBlock * sizeof(float)
	size_t sharedBytes = threadsPerBlock * sizeof(float);
    dotProductGPU<<<numBlocks, threadsPerBlock, sharedBytes>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, 1*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	/// final reduction on CPU over numBlocks partial sums
    DotGPU = 0.0f; // C_GPU was copied into C_CPU.
    for (int i=0; i<numBlocks; ++i)
    {
        DotGPU +=C_CPU[i];
    }
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp(numBlocks);	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
