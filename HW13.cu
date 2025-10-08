// Name:
// Page-locked memory test
// nvcc 13PageLockedMemory.cu -o temp

/*
This program tests the speed difference between using pageable memory 
(default type of memory; its contents can be transferred from RAM to other secondary 
storage locations by the Operating System) vs page-locked memory (explicitly locked in physical RAM,
not allowed to be moved to secondary storage by the Operating System).
when transferring data between the CPU and GPU.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define SIZE 2000000 
#define NUMBER_OF_COPIES 1000

//Globals
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
cudaEvent_t StartEvent, StopEvent;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void cleanUp();
void copyPageableMemoryUp();
void copyPageLockedMemoryUp();
void copyPageableMemoryDown();
void copyPageLockedMemoryDown();

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&NumbersOnGPU, SIZE*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate pageable Host (CPU) Memory
	PageableNumbersOnCPU = (float*)malloc(SIZE*sizeof(float));
	
	//Allocate page locked Host (CPU) Memory
    /*
    Passes in the address of PageLockedNumbersOnCPU to the function so that it can modify
    the pointer to point to the newly allocated page-locked memory.

    SIZE*sizeof(float) --> 2,000,000 floats (8,000,000 bytes)

    This is different from malloc() because it creates page-locked memory which 
    stays in physical RAM; this allows the GPU direct memory access to transfer data without 
    interference from the OS.  
    */
	cudaMallocHost(&PageLockedNumbersOnCPU, SIZE*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(NumbersOnGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
    /*
    Deallocates the memory that was allocated for page-locked memory on the host.

    You cannot use free() on page-locked memory; you must use cudaFreeHost().
    
    */
	cudaFreeHost(PageLockedNumbersOnCPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	free(PageableNumbersOnCPU); 
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

void copyPageableMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageableMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

/*
Copies data from the page-locked memory on the host (PageLockedNumbersOnCPU) 
to the device memory (NumbersOnGPU)
Transfers a number of bytes equal to SIZE * sizeof(float) (8,000,000 bytes).

cudaMemcpyHostToDevice indicates the direction; host ---> device
Tests page-locked memory upload performance; this function is indentical to copyPageableMemoryUp()
except it uses page-locked memory instead of pageable memory, meaning the GPU can use 
Direct Memory Access (DMA) to transfer the data without OS interference.
Allows for comparision of transfer speeds between pageable and page-locked memory.

*/
void copyPageLockedMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		
        cudaMemcpy(NumbersOnGPU, PageLockedNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

/*
This function tests page-locked memory download performance (GPU --> CPU).
At this point, each of the following comparisons have been done: 
1. Pageable memory upload (CPU --> GPU)
2. Page-locked memory upload (CPU --> GPU)
3. Pageable memory download (GPU --> CPU)
4. Page-locked memory download (GPU --> CPU)
*/
void copyPageLockedMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		//Same cudaMemcpy call as just above, but the direction is reversed. 
        cudaMemcpy(PageLockedNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory down = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory down = %3.1f milliseconds", timeEvent);
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
