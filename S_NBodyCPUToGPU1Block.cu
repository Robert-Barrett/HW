// Name: Robert Barrett 
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc S_NBodyCPUToGPU1Block.cu.cu -o temp -lglut -lm -lGLU -lGL
/*
Additions and changes of note in this program:
(1) Minimize data transfers from the CPU to the GPU. These transfers are taxing on performance, so keeping the data on the GPU 
    duirng computation, and only transferring back to the CPU for visuals is key. 
(2) Advantages of doing this work in parallel. With each timestep doing N(N + 1) operations, doing this in parallel compared 
    to sequentially makes the GPU version of the code run much faster than the CPU version.  
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// CPU Memory - used by OpenGL for drawing 
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float GlobeRadius, Diameter, Radius;
float Damp;

// GPU memory - used for the heavy computational work
float3 *d_P, *d_V, d_F;
float *d_M;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
int main(int, char**);

/*
(1a) Each of N threads calculates forces acting on 1 body. Thread 0 ---> Body 0, Thread 1 ---> Body 1, etc. 
(2) Each thread computes forces on body i from all other bodies j
(3) Accumulate forces on body i; Using atomicAdd to prevent race conditions
*/  
__global__ void computeForces(float3 *P, float3 *F, float *M, int N)
{
	int i = threadIdx.x;
	//(1)
	if(i < N)
	{
		float force_mag;
		float dx, dy, dz, d, d2;
		//(2)
		for(int j = 0; j < N; j++)
		{
			if(i != j)
			{
				dx = P[j].x - P[i].x;
				dy = P[j].y - P[i].y;
				dz = P[j].z - P[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d = sqrt(d2);
				
				// Lennard-Jones type force
				force_mag = (G*M[i]*M[j])/(d2) - (H*M[i]*M[j])/(d2*d2);
				
				// (3)
				atomicAdd(&F[i].x, force_mag*dx/d);
				atomicAdd(&F[i].y, force_mag*dy/d);
				atomicAdd(&F[i].z, force_mag*dz/d);
			}
		}
	}
}

/*
Each thread updates on body independently.
*/
__global__ void integrateMotion(float3 *P, float3 *V, float3 *F, float *M, float dt, float time, float Damp, int N)
{
	int i = threadIdx.x;
	
	if(i < N)
	{
		// Update velocity
		if(time == 0.0)
		{
			V[i].x += (F[i].x/M[i])*0.5*dt;
			V[i].y += (F[i].y/M[i])*0.5*dt;
			V[i].z += (F[i].z/M[i])*0.5*dt;
		}
		else
		{
			V[i].x += ((F[i].x - Damp*V[i].x)/M[i])*dt;
			V[i].y += ((F[i].y - Damp*V[i].y)/M[i])*dt;
			V[i].z += ((F[i].z - Damp*V[i].z)/M[i])*dt;
		}
		
		// Update position
		P[i].x += V[i].x*dt;
		P[i].y += V[i].y*dt;
		P[i].z += V[i].z*dt;
	}
}

/*
Instead of serially iterating through a loop, all forces are reset to 0 in parallel.
Thread 0 ---> F[0], Thread 1 ---> F[1], and so on. 
*/
__global__ void resetForces(float3 *F, int N)
{
	int i = threadIdx.x;
	
	if(i < N)
	{
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
	}
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}
/*
(1) Allocates memory for N bodies with position (P), velocity (V), force (F), and mass (M). 
(2) Calculates the diamter where the Lennard-Jones force equals 0 
(3a) Creates a large 'globe'; setting the initial velosity, inotial force, and mass.
(3b) Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
(3c) Making sure they are atleast one diameter apart. If they are not throw these positions away and try again.
(4) Allocate DEVICE (GPU) memory
(5) Copy initial data from HOST to DEVICE
*/
void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	Damp = 0.5;
    	
    	// (1)
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	//(2)
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); 
	Radius = Diameter/2.0;
	/*
   (3a)
   Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	 Then we double this radius just so we can get all the bodies setup with no problems. 
  */
	
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	//(3b)
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// (3c)
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	
	// (4)
	cudaMalloc(&d_P, N*sizeof(float3));
	cudaMalloc(&d_V, N*sizeof(float3));
	cudaMalloc(&d_F, N*sizeof(float3));
	cudaMalloc(&d_M, N*sizeof(float));
	
	// (5)
	cudaMemcpy(d_P, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, N*sizeof(float), cudaMemcpyHostToDevice);
	
	printf("\n To start timing type s.\n");
}



/*
Each Kenrel launches 1 block of N threads. 
(1) Reset forces to zero on GPU
(2) Compute forces on GPU
(3) Integrate motion on GPU
(4) Copy positions back to host for drawing
(5) Copy final results back to host
*/
void nBody()
{
	int    drawCount = 0; 
	float  time = 0.0;
	float dt = 0.0001;

	while(time < RUN_TIME)
	{
		// (1)
		resetForces<<<1, N>>>(d_F, N);
		cudaDeviceSynchronize();
		
		// (2)
		computeForces<<<1, N>>>(d_P, d_F, d_M, N);
		cudaDeviceSynchronize();
		
		// (3)
		integrateMotion<<<1, N>>>(d_P, d_V, d_F, d_M, dt, time, Damp, N);
		cudaDeviceSynchronize();

		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{
				// (4)
				cudaMemcpy(P, d_P, N*sizeof(float3), cudaMemcpyDeviceToHost);
				drawPicture();
			}
			drawCount = 0;
		}
		
		time += dt;
		drawCount++;
	}
	
	// (5)
	cudaMemcpy(P, d_P, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(F, d_F, N*sizeof(float3), cudaMemcpyDeviceToHost);
}


int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}




