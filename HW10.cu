// Name: Robert Barrett 
// Robust Vector Dot product 
// nvcc HW10.cu -o temp

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 564123 // Length of the vector
#define BLOCK_SIZE 1024 // Threads in a block

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int N_padded; // vectors new size w/ the added 0's
float *results_GPU; // hold the final answer on the GPU

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();
void checkGPU();
void calculatePadding();

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
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = N_padded / BlockSize.x; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;

    printf("Grid size: %d blocks\n", GridSize.x);
    printf("Each block: %d threads\n", BlockSize.x);
}

// Allocating the memory we will be using, which now depends on the new, padded, size of the vector, and one final spot for a resultant vector 
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N_padded*sizeof(float));
	B_CPU = (float*)malloc(N_padded*sizeof(float));
	C_CPU = (float*)malloc(N_padded*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N_padded*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N_padded*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N_padded*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&result_GPU, sizeof(float));
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

    if(N_padded > N)
    {
        printf("Setting padded elements to 0 ... \n");
        memeset(&A_GPU[N],0,(N_padded-N)*sizeof(float));
        memeset(&B_GPU[N],0,(N_padded-N)*sizeof(float));
    }
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *result, int n)
{
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
// if statement is new 
    if(vectorIndex < n)
    {
        c_sh[threadIndex] = a[vectorIndex] * b[vectorIndex];
    }
    else
    {
        c_sh[threadIndex] = 0.0f;
    }
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)
	{
		if(fold%2 != 0)
		{
			if(threadIndex == 0 && (vectorIndex + fold - 1) < n)
			{
				c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(threadIndex < fold && (vectorIndex + fold) < n)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
	c[blockDim.x*blockIdx.x] = c_sh[0];
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
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

/*
REQUIREMENTS: Compute Capability at least 3 (for atomic add), and block and grid size within limits. 

checkGPU() is a algorithm which is both validating (it checks each GPUs properties with respect to the programs requirements, and skips any GPU which doesn't meet them)
and selecting (among the GPUs which fit the criteria, it picks the best one for the job)

cudaDeviceProp is a special data type which essentially serves as form the GPU fills out. It holds thigs such as the GPUs name, version number (major), sub version (minor),
max threads per block, and how much memory the GPU has. This variable was created so we can compare these features if the machine contains more than one GPU

(1) cudaGetDeviceCount() counts available GPUs and stores it in numGPUs address; the result is then printed to the user 
(2) Enters a for loop which checks each GPUs hardware specifications using cudaGetDeviceProperties(). In the for loop, the validation checks are done in order of
importance 
    (2a) compute capablity defined whether or not certain CUDA features are supported by the hardware  
    (2b) Block size is the next most limiting factor because if the blocks are too big, the kernel would fail to launch 
    (2c) Grid size is the least restricing because most GPUs can handle large grid sizes, but it is still finite 
(3) cudaSetDevice() tells CUDA which GPU to use for all operations. This needs to happen before any memeory allocation (because it needs to know which GPU to allocate on)
and before any kernel launches (CUDA needs to know which GPU to launch the kernel on)
(4) Using bestGPU as an index which points to whicever GPU is the best so far, cudaGetDeviceProperties(&bestInfo, bestGPU) fills in the information into bestInfo
the if statement the compares 2 structures info (the information about the GPU currently in the loop) and bestInfo (the record holder). The if statement checks if 
one GPU has a higher info.major than the other, and if it does that one wins by default; if the majors are equal, it then checks the info.minor for each  
*/
void checkGPU()
{
    // (1)
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    printf("Number of GPUs found: %d\n", numGPUs);
    
    // If no GPUs, quit
    if (numGPUs == 0) {
        printf("ERROR: No GPUs found!\n");
        exit(1);
    }
    //(2)
    int bestGPU = 0;  // Start by assuming GPU 0 is best (rememeber GPUs are indexed like threads or blocks)
    
    for (int i = 0; i < numGPUs; i++) 
    {
        cudaDeviceProp info;
        cudaGetDeviceProperties(&info, i);
        
        printf("\nGPU %d: %s\n", i, info.name);
        
        // (2a)
        if (info.major < 3) {
            printf("  - Too old (version %d.%d, need 3.0+)\n", info.major, info.minor);
            continue;  // If a GPU fails this test, it won't even bother going through the lest, showing earlier tests are more important 
            exit(1);
        }
        printf("  - Version %d.%d \n", info.major, info.minor);
        
        // (2b)
        if (BLOCK_SIZE > info.maxThreadsPerBlock) {
            printf("  - Block size %d too big (max is %d)\n", BLOCK_SIZE, info.maxThreadsPerBlock);
            continue;  
            exit(1);
        }
        printf("  - Block size %d fits \n", BLOCK_SIZE);
        
        // (2c)
        int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Same as GridSize.x
        if (gridSize > info.maxGridSize[0]) //info.maxGridSize[0] only checks in the x-direction 
        {
            printf("  - Grid size %d too big (max is %d)\n", gridSize, info.maxGridSize[0]);
            continue;  
            exit(1);
        }
        printf("  - Grid size %d fits \n", gridSize);
        
        
        printf("  - This GPU works! \n");
        
        // (4)
        cudaDeviceProp bestInfo;
        cudaGetDeviceProperties(&bestInfo, bestGPU);
        
        
        if (info.major > bestInfo.major or
            (info.major == bestInfo.major and info.minor > bestInfo.minor)) {
            bestGPU = i;
            printf("  - This is now our best GPU!\n");
        }
    }  
    // (3)
    cudaSetDevice(bestGPU);
    
    cudaDeviceProp finalInfo;
    cudaGetDeviceProperties(&finalInfo, bestGPU);
    printf("\nUsing GPU %d: %s (version %d.%d)\n",bestGPU, finalInfo.name, finalInfo.major, finalInfo.minor);
}
/*
(1) Uses int leftover to decide whether or not the vector is cleanly divisble into the blocks
(2) If there aren't leftoever, the vector stays the same. If there are leftovers, the number of 0's to add is the difference between block size and the leftovers
*/
void calculatePadding()
{
   //(1)
    int leftover = N % BLOCK_SIZE;
    //(2)
    if (leftover == 0) 
    {
        N_padded = N;
        printf("Vector fits perfectly - no padding needed\n");
    } 
    else 
    {
        int zeros_to_add = BLOCK_SIZE - leftover;
        N_padded = N + zeros_to_add;
        printf("Adding %d zeros to make vector fit perfectly\n", zeros_to_add);
        printf("New vector size: %d (was %d)\n", N_padded, N);
    }
}

int main()
{
    checkGPU();
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	DotGPU = 0.0;
	for(int i = 0; i < N; i += BlockSize.x)
	{
		DotGPU += C_CPU[i]; // C_GPU was copied into C_CPU. 
	}

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
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}

