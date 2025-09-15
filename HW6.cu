// Name: Robert Barrett 
// Simple Julia GPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.

#include <stdio.h>
#include <GL/glut.h>
//provices access to CUDA runtime API
#include <cuda_runtime.h>

// Defines
#define MAXMAG 10.0
#define MAXITERATIONS 200
#define A  -0.824
#define B  -0.1711

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;


/*
Now that we're doing this operation on the GPU, each thread needs to know which pixel it's going to work on. 
We compute the coordinates of each pixel (i,j) from blockIDx and threadIdx
Removed the loop (while (y<YMax)) and (while (x<XMax)) since each thread is now responsible for a single pixel, and the GPU can do this in parallel.
*/
__global__ void escapeOrNotColorKernel(float *pixels, float XMin, float YMin, float stepSizeX, float stepSizeY,int width, int height) 
{
    /* Imagine the grid as a theater; each block is a section and each thread is a seat in that section. Each seat needs
    so the formula is saying the seat number = (section number * seats per section) + seat number in that section
    For 2 dimensions, we do this for both x and y directions.

    For example a grid of 2x2 blocks and each block will have 4x4 threads (meaning the image is 8x8 pixels, 64 in total)
    So the full image size would be:
    gridDim.x = 2 = gridDim.y
    blockDim.x = 4 = blockDim.y
    Each thread in a block in indexed (0,0) to (3,3) 
    Each block in indexed (0,0) to (1,1)
    Combining the block and thread position gives 
    i = blockIdx.x * blockDim.x + threadIdx.x
    j = blockIdx.y * blockDim.y + threadIdx.y

    Ex: 
    Suppose in a 2x2 grid, I'm on the top right block, block (1,0)
    blockIdx.x = 1, blockIdx.y = 0, blockDim.x = 4
    Take a thread inside this block, say (2,3) = (threadIdx.x, threadIdx.y)
    Plugging into the formula for i and j
    i = (1 * 4) +2 = 6
    j = (0 * 4) +3 = 3
    So this thread is responsible for pixel (6,3) in the full image
    Drawing the 8x8 image and labeling the pixels (i,j), each block would fill its own 4x4 corner of the image, ensuring no overlap or gaps. 
    (0,0) (1,0) (2,0) (3,0) | (4,0) (5,0) (6,0) (7,0)
    (0,1) (1,1) (2,1) (3,1) | (4,1) (5,1) (6,1) (7,1)
    (0,2) (1,2) (2,2) (3,2) | (4,2) (5,2) (6,2) (7,2)
    (0,3) (1,3) (2,3) (3,3) | (4,3) (5,3) (6,3) (7,3)
    ----------------------- + -----------------------
    (0,4) (1,4) (2,4) (3,4) | (4,4) (5,4) (6,4) (7,4)
    (0,5) (1,5) (2,5) (3,5) | (4,5) (5,5) (6,5) (7,5)
    (0,6) (1,6) (2,6) (3,6) | (4,6) (5,6) (6,6) (7,6)
    (0,7) (1,7) (2,7) (3,7) | (4,7) (5,7) (6,7) (7,7)
    

    */
    int i = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // pixel y
    // if the sqaure is outside of the mural's area, the painter takes a smoke break
    if (i >= width or j >= height) return;
    //converts each pixel (i,j) into coordinate points (x,y) on the complex plane 
    float x = XMin + i * stepSizeX;
    float y = YMin + j * stepSizeY;

    float mag, tempX;
    int count = 0;
    mag = sqrtf(x*x + y*y);
    // applying the formula repeatedly, finding the distance from the origin, and deciding whether or not that point escapes or is trapped. 
    while (mag < MAXMAG and count < MAXITERATIONS) {
        tempX = x;
        x = x*x - y*y + A;
        y = (2.0f * tempX * y) + B;
        mag = sqrtf(x*x + y*y);
        count++;
    }

float red = 0.0f;
float green = 0.0f;
float blue = 0.0f;
if (count >= MAXITERATIONS)
{
    red = 1.0f; // colors trapped points red
}
/*
Instead of returning a float like the CPU did, the GPU kernel writes 
the color value into the appropriate location in the pixels array.

We have a 2d image that has dimensions width x height. 
in memmory, this is represented with a 1-dimensional array of numbers 
Each pixel has 3 values. So the array for a 3x2 image for example would look like 
pixels = [R0,G0,B0,   R1,G1,B1,   R2,G2,B2,   R3,G3,B3,   R4,G4,B4,   R5,G5,B5]
Since each pixel takes up three slots of the array, we multiply the pixel index by 3 to get the right spot in the array.
Ex: Suppose the image has width = 4, and I want to pixel on the 2nd column of the 1st row (2,1). Applying the formula:
pixelNumber = (j * width + i) = 1*4 + 2 = 6, and accounting for fact RGB takes up 3 slots in the array, multiply by 3:
idx = pixelNumber * 3 = 6 * 3 = 18. So the red value for pixel (2,1) is stored in pixels[18], green in pixels[19], and blue in pixels[20].
*/
int idx = (j * width + i) * 3;
pixels[idx]   = red; 
pixels[idx+1] = green;  // green
pixels[idx+2] = blue;  // blue
}

void display(void) 
{ 
 // Using cudaMalloc, cudaMemcpy, and cudaFree to manage memory on the GPU.
 
    float *pixels; 
    float *d_pixels;
//  Allocate memory on the CPU and GPU

    pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
    cudaMalloc((void**)&d_pixels, WindowWidth*WindowHeight*3*sizeof(float));

// how much to move each step in the x and y direction on the complex plane

    float stepSizeX = (XMax - XMin)/((float)WindowWidth);
    float stepSizeY = (YMax - YMin)/((float)WindowHeight);

// grid is sized such that each pixel (i,j) is handled by a single thread.

    dim3 block(16,16);
    dim3 grid((WindowWidth+block.x-1)/block.x, (WindowHeight+block.y-1)/block.y);
    escapeOrNotColorKernel<<<grid, block>>>(d_pixels, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);

// Wait for GPU to finish before accessing on host

    cudaDeviceSynchronize();

// Copy result back to host
    cudaMemcpy(pixels, d_pixels, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);

// Put pixels on the screen
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
    glFlush(); 

// Cleaning up our room 
    cudaFree(d_pixels);
    free(pixels);
}

int main(int argc, char** argv) { 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(WindowWidth, WindowHeight);
    glutCreateWindow("Fractals--Man--Fractals");
    glutDisplayFunc(display);
    glutMainLoop();
}
