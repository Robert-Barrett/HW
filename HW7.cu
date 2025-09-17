// Name: Robert Barrett 
// Flexible Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL -lm
/*
Suggested parameters for cool fractals:
A = -0.70176 B = 0.3842 MaxIterations = 500
--------------------------------------------
A = -0.8 B = 0.156 MaxIterations = 300
--------------------------------------------
A = -0.7269 B = 0.1889 MaxIterations = 800 
*/
#include <stdio.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <math.h>


#define MAXMAG 10.0
int MaxIterations = 200;
float A = 0.57904f;  
float B = -0.2478f;  


unsigned int WindowWidth  = 1000;   
unsigned int WindowHeight = 1000;   

float XMin = -2.0f, XMax = 2.0f;
float YMin = -2.0f, YMax = 2.0f;

void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, 
                            int width, int height, float A, float B, int MaxIterations);
void display(void);

void cudaErrorCheck(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n",
               cudaGetErrorString(error), file, line);
        exit(0);
    }
}
/*
Each thread calculates the pixel's corresponding location on the complex plane, then applies 
the Julia set formula until it's distance is larger than the MAXMAG or hits the MaxIterations. 
Assigns a color based on how fast the pixel escapes; then the function writes those color 
values into a shared pixel buffer, which is then copied back and drawn to the screen. 
-------------------------------------------------------------------------------------------------
Maps (i,j) --> (x,y) using the step sizes dx and dy
x = Xmin + i * dx | y = Ymin + j * dy
For each pixel's (x,y), treat is as a complex number z = x + i*y, 
then iterate z --> z*z + C, C = A + Bi
If |z| > MAXMAG, the point escapes, if it doesn't escape before reaching MaxIterations, we assume it's trapped
*/
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy,
                            int width, int height, float A, float B, int MaxIterations) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // pixel x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // pixel y
// Ensures that threads that fall outside the image bounds quit early 
    if (i >= width or j >= height) return;
    int id = 3 * (j * width + i);

    float x = xMin + dx * i;
    float y = yMin + dy * j;

    float mag, tempX;
    int count = 0;

    mag = sqrtf(x*x + y*y);
    while (mag < MAXMAG and count < MaxIterations) 
    {
        tempX = x;
        x = x*x - y*y + A;
        y = (2.0f * tempX * y) + B;
        mag = sqrtf(x*x + y*y);
        count++;
    }

    
    float t = (float)count / MaxIterations;
    if (count == MaxIterations)
    {
        pixels[id] = 0.0f;
        pixels[id+1] = 0.0f;
        pixels[id+2] = 0.0f;
    } 
    else
    {
        pixels[id] = fminf(1.0f, 3.0f * t);
        pixels[id+1] = fminf(1.0f, 3.0f * (t - 0.33f));
        pixels[id+2] = fminf(1.0f, 3.0f * (t - 0.66f));
    }
   
}
/*
Called by OpenGl when the window needs to be redrawn. Sets up memory for storing pixel colors 
both on the CPU and GPU, then the step sizes that map pixels to points in the fractal plane.
Then, it launches the colorPixels kernel across a grid of GPU threads s.t each pixel is computed in 
parallel. Once the GPU is finished, thew pixel data is copied back to the CPU and drawn on the screen with 
glDrawPixels. Then it frees the memory on the GPU and CPu  
*/
void display(void) {
    float *pixelsCPU, *pixelsGPU; 
    float stepSizeX = (XMax - XMin) / ((float)WindowWidth);
    float stepSizeY = (YMax - YMin) / ((float)WindowHeight);

    pixelsCPU = (float*)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaMalloc(&pixelsGPU, WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);

    dim3 blockSize(16, 16);
    dim3 gridSize((WindowWidth + blockSize.x - 1) / blockSize.x,
                  (WindowHeight + blockSize.y - 1) / blockSize.y);

    colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin,
                                         stepSizeX, stepSizeY,
                                         WindowWidth, WindowHeight, A, B, MaxIterations);
    cudaErrorCheck(__FILE__, __LINE__);

    cudaMemcpy(pixelsCPU, pixelsGPU,
               WindowWidth * WindowHeight * 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
    glFlush();

    cudaFree(pixelsGPU);
    free(pixelsCPU);
}
/*
Reads any command-line arguments provided by the user to set the window size, fractal parameters 
A and B, and the number of iterations. Initializes the graphics library (GLUT), creates the display window
and registers display() as the function that will handle drawing. 
*/
int main(int argc, char** argv) 
{
    if (argc >=3)
    {
        WindowWidth = atoi(argv[1]);
        WindowHeight = atoi(argv[2]);
    }
    if (argc >= 5)
    {
        A = atof(argv[3]);
        B = atof(argv[4]);
    }
    if (argc >= 6)
    {
        MaxIterations = atoi(argv[5]);
    }
    
    printf("Window size: %u x %u | C = %.4f + %.4fi | MaxIterations = %d\n", 
                            WindowWidth, WindowHeight, A, B, MaxIterations);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(WindowWidth, WindowHeight);
    glutCreateWindow("GPU Julia Set");
    glutDisplayFunc(display);
    glutMainLoop();
}
