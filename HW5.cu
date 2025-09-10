// Name: Robert Barrrett
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
    // Ask Cuda how many GPUs are connected
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
    //loops over each GPU and prints its properties
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__); 
		printf(" ---General Information for device %d ---\n", i);
        //Prints a reedable name for the GPUs
		printf("Name: %s\n", prop.name);
        // Compute capability == GPU's architecture version, which determines the CUDA features supported by the GPU
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        //GPU's clock frequency in kilohertz. Bigger number == faster operations
		printf("Clock rate: %d\n", prop.clockRate);
		// the next three lines tells you if the GPU can transfer memory from the device to the host at the same time it is executing a kernel
        printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
        //  may have a watchdog timer that will reset the GPU if a kernel takes too long to execute
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
        //Total GPU memeory in bytes (VRAM == Video random access memory)
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
        //Total constant memory in bytes
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
        // pitch == the width in bytes of the allocation
		printf("Max mem pitch: %ld\n", prop.memPitch);
        // Required byte alignment for textures (tells you how much data must be aligned in memory for optimal access)
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" ---MP Information for device %d ---\n", i);
        //Number of streaming multiprocessors (SMs) on the GPU
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); 
        //Amount of shared memory available per block in bytes (shared memory is shared between threads in a block)
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        //Max number of 32-bit registers available per block (registers are the fastest form of memory)
		printf("Registers per mp: %d\n", prop.regsPerBlock);
        // number of threads which can exceute the same instruction at the same time (in lockstep)
		printf("Threads in warp: %d\n", prop.warpSize);
        // max number of threads that can be launched in a single block
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        //Max threads per block in each dimension (x,y,z)
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        //Max block per grid in each dimension (x,y,z)
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		
        //extra things not printed if the original code

        // the frequency (in kilohertz) at which the memory on the GPU operates
        printf("Memory clock rate (kHz): %d\n", prop.memoryClockRate);
        // the width (in bits) of the memory bus that connects the GPU to its memory; wider bus == more data can be transferred per clock cycle
        printf("Memory bus width (bits): %d\n", prop.memoryBusWidth);
        // Size of the GPUs Level 2 cache in bytes, more cache reduces latency when repeatedly accessing data from global memory
        printf("L2 cache size (bytes): %d\n", prop.l2CacheSize);
        // Maximum number of threads that can be active on a single multiprocessor at the same time
        printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        // IF enabled, the GPU and CPU share a unified address space, which simplifies memory management; they both have access to the same memory locations
        printf("Unified addressing supported: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        // If the returns YES, the GPU can run multiple kernels at once 
        printf("Concurrent kernels supported: %s\n", prop.concurrentKernels ? "Yes" : "No");
        // Number of asynchronous engines supported by the GPU, asynchronous engines handle data transfers between the GPU and host memory without 
        // blocking the CPU and GPU from performing other tasks
        printf("Async engine count: %d\n", prop.asyncEngineCount);
        // If this returns YES, the GPU shares system memory instead of having dedicated VRAM
        printf("Integrated GPU (uses system RAM): %s\n", prop.integrated ? "Yes" : "No");
        // If this returns YES, the GPU can access host memory, allowing for faster data transfers between the CPU and GPU
        printf("Can map host memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        // Shows if Error-Correcting Code is turned on, which helps detect and correct memory errors (memory error == when data is recalled incorrectly)
        printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        // Tells you where the GPU is physically located on the PCI bus (PCI bus == a high-speed interface that connects the GPU to the CPU 
        //and other components in the system)
        printf("PCI Bus ID: %d\n", prop.pciBusID);
        printf("PCI Device ID: %d\n", prop.pciDeviceID);
        // Tells if the GPU is running on TCC (Tesla Compute CLuster) which disables graphics-related features to optimize the GPU for compute tasks
        printf("TCC driver enabled: %s\n", prop.tccDriver ? "Yes" : "No");

        // Texture and surface limits
        printf(" ---Texture and Surface Information for device %d ---\n", i);
        // Max length of 1D textures
        printf("Max 1D texture size: %d\n", prop.maxTexture1D);
        // Max dimensions of 2D textures (width, height)
        printf("Max 2D texture dimensions: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
        // Max dimensions of 3D textures (width, height, depth)
        printf("Max 3D texture dimensions: (%d, %d, %d)\n",
            prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
            // Max resolution of cubemap textures (used for environment mapping) A cubemap texture consists of six square textures that represent the faces of a cube
        printf("Max cubemap texture size: %d\n", prop.maxTextureCubemap);
        // Max dimensions of 2D layered textures (width, height, number of layers)
        printf("Max 2D layered texture dimensions: (%d, %d), layers: %d\n",
            prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
        // Max length of 1D surfaces (surfaces memory is similar to texture but it's writable) 
        printf("Max 1D surface size: %d\n", prop.maxSurface1D);
        // Max dimensions of 2D surfaces (width, height)
        printf("Max 2D surface dimensions: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
        // Max dimensions of 3D surfaces (width, height, depth)                                                                        
        printf("Max 3D surface dimensions: (%d, %d, %d)\n",
                 prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);
        
        printf("\n");
	}	
	return(0);
}
