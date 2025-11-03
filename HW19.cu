// Name:
// GPU random walk. 
// nvcc HW19.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
// GPU RNG functions 
#include <curand_kernel.h>

// Defines
#define NUM_WALKS 20
// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
__global__ void randomWalkKernel(int *positionsX, int *positionsY, int numSteps); 
int main(int, char**);
// (1) Generates a number in the range [0, RAND_MAX]; the initial position is (0,0)
// (2) Direction of movement depends on whether or not
//	the value is less than the midpoint; if so, move in the neg direction.
//	if not, move in the positive direction. 
// (3) Print your final position 


int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

// (1) Each thread calculates its own ID 
// (2) Each walk gets its own RNG with a different seed, ensuring each walk is actually different 
// (3) the for loop uses similar logic to the CPU's loop. The difference is 
// the use of curand_uniform. This returns a float between 0 and 1, and we 
// take numbers less than 0.5 to go negative, and numbers greater than to go positive
__global__ void randomWalkKernel(int *positionsX, int *positionsY, int numSteps)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < NUM_WALKS)
	{
		curandState state;
		curand_init(1234 + idx, 0, 0, &state);
		
		int posX = 0;
		int posY = 0;
		
		for(int i = 0; i < numSteps; i++)
		{
			float randX = curand_uniform(&state);
			float randY = curand_uniform(&state);
			
			posX += (randX < 0.5f) ? -1 : 1;
			posY += (randY < 0.5f) ? -1 : 1;
		}
		
		positionsX[idx] = posX;
		positionsY[idx] = posY;
	}
}

int main(int argc, char** argv)
{
	srand(time(NULL));
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	
	printf("\n Final position = (%d,%d) \n", positionX, positionY);

	// GPU version
	int *positionsX, *positionsY;
	// Creating arrays both the GPU and CPU can see, so now manual copying is needed 
	cudaMallocManaged(&positionsX, NUM_WALKS * sizeof(int));
	cudaMallocManaged(&positionsY, NUM_WALKS * sizeof(int));
	
	int threadsPerBlock = 256;
	int blocks = (NUM_WALKS + threadsPerBlock - 1) / threadsPerBlock;
	
	randomWalkKernel<<<blocks, threadsPerBlock>>>(positionsX, positionsY, NumberOfRandomSteps);
	cudaDeviceSynchronize();
	
	printf("\n GPU Final positions (20 random walks):\n");
	for(int i = 0; i < NUM_WALKS; i++)
	{
		printf(" Walk %2d: (%d, %d)\n", i, positionsX[i], positionsY[i]);
	}
	
	cudaFree(positionsX);
	cudaFree(positionsY);
	
	return 0;

}
