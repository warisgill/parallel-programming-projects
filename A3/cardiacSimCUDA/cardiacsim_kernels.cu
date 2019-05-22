/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 * Refer to "Detailed Numerical Analyses of the Aliev-Panfilov Model on GPGPU"
 * https://www.simula.no/publications/detailed-numerical-analyses-aliev-panfilov-model-gpgpu
 * by Xing Cai, Didem Unat and Scott Baden
 *
*/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>
using namespace std;

#define TILE_DIM 16

// For Command Line Args
void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads);
// Timer: Make successive calls and take a difference to get the elapsed time.
double getTime();
// Allocate a 2D array
double **alloc2D(int m, int n);
// Mirror Ghost Boundries
void mirrorBoundries(double *E_prev_1D, const int n, const int m, const int WIDTH);
/* 
	Reports statistics about the computation
	These values should not vary (except to within roundoff)
 	when we use different numbers of  processes to solve the problem
*/
double stats(double **E, int m, int n, double *_mx);
double stats1D(double *E, int m, int n, double *_mx, int WIDTH);


// ============================== Kernels  ===========================
__global__ void simulate_version1_PDE(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH)
{

	int RADIUS = 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;

	if (row >= 1 && row <= m && col >= 1 && col <= n)
	{
		E_1D[row * WIDTH + col] = E_prev_1D[row * WIDTH + col] + alpha * (E_prev_1D[row * WIDTH + (col + 1)] + E_prev_1D[row * WIDTH + (col - 1)] - 4 * E_prev_1D[row * WIDTH + col] + E_prev_1D[(row + 1) * WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);
	}
}

__global__ void simulate_version1_ODE(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH)
{
	int RADIUS = 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int index = row * WIDTH + col;

	if (row >= 1 && row <= m && col >= 1 && col <= n)
	{
		E_1D[index] = E_1D[index] - dt * (kk * E_1D[index] * (E_1D[index] - a) * (E_1D[index] - 1) + E_1D[index] * R_1D[index]);
		R_1D[index] = R_1D[index] + dt * (epsilon + M1 * R_1D[index] / (E_1D[index] + M2)) * (-R_1D[index] - kk * E_1D[index] * (E_1D[index] - b - 1));
	}
}

__global__ void simulate_version2(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH)
{
	int RADIUS = 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;

	int index = row * WIDTH + col;

	if (row >= 1 && row <= m && col >= 1 && col <= n)
	{

		// PDE
		E_1D[row * WIDTH + col] = E_prev_1D[row * WIDTH + col] + alpha * (E_prev_1D[row * WIDTH + (col + 1)] + E_prev_1D[row * WIDTH + (col - 1)] - 4 * E_prev_1D[row * WIDTH + col] + E_prev_1D[(row + 1) * WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);

		//ODE
		E_1D[index] = E_1D[index] - dt * (kk * E_1D[index] * (E_1D[index] - a) * (E_1D[index] - 1) + E_1D[index] * R_1D[index]);
		R_1D[index] = R_1D[index] + dt * (epsilon + M1 * R_1D[index] / (E_1D[index] + M2)) * (-R_1D[index] - kk * E_1D[index] * (E_1D[index] - b - 1));
	}
}

__global__ void simulate_version3(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH)
{
	int RADIUS = 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int index = row * WIDTH + col;

	if (row >= 1 && row <= m && col >= 1 && col <= n)
	{
		// PDE
		E_1D[row * WIDTH + col] = E_prev_1D[row * WIDTH + col] + alpha * (E_prev_1D[row * WIDTH + (col + 1)] + E_prev_1D[row * WIDTH + (col - 1)] - 4 * E_prev_1D[row * WIDTH + col] + E_prev_1D[(row + 1) * WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);

		double e_temp = E_1D[index];
		double r_temp = R_1D[index];

		//ODE
		e_temp = e_temp - dt * (kk * e_temp * (e_temp - a) * (e_temp - 1) + e_temp * r_temp);
		r_temp = r_temp + dt * (epsilon + M1 * r_temp / (e_temp + M2)) * (-r_temp - kk * e_temp * (e_temp - b - 1));

		E_1D[index] = e_temp;
		R_1D[index] = r_temp;
	}
}

__global__ void simulate_version4(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH)
{
	// Shared Memory Allocation
	__shared__ double tempR[TILE_DIM + 2][TILE_DIM + 2];
	__shared__ double tempE[TILE_DIM + 2][TILE_DIM + 2];
	__shared__ double tempE_prev[TILE_DIM + 2][TILE_DIM + 2];
	
	int RADIUS = 1;
	// Global Indexing
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int index = row * WIDTH + col;
	
	// Local or Shared Indexing
	int l_row = threadIdx.y + RADIUS;
	int l_col = threadIdx.x + RADIUS;
	
	if (row >= 1 && row <= m && col >= 1 && col <= n ){
		tempR[l_row][l_col] = R_1D[index];
		tempE_prev[l_row][l_col] = E_prev_1D[index];
	}

	// copy left 
	if (l_col == 1){
		tempE_prev[l_row][l_col-1] = E_prev_1D[index-1];
	}

	// copy Right 
	if (l_col == TILE_DIM){
		tempE_prev[l_row][l_col+1] = E_prev_1D[index+1];
	}

	// copy up
	if (l_row == 1){
		tempE_prev[l_row-1][l_col] = E_prev_1D[index - WIDTH];
	}

	// copy down
	if (l_row == TILE_DIM){
		tempE_prev[l_row+1][l_col] = E_prev_1D[index + WIDTH];
	}

	// Make sure all threads get to this point before proceeding!
	__syncthreads(); // This will syncronize threads in a block

	if (row >= 1 && row <= m && col >= 1 && col <= n)
	{	
		// PDE v1
		//tempE[l_row][l_col] = E_prev_1D[index] + alpha * (E_prev_1D[row * WIDTH + (col + 1)] + E_prev_1D[row * WIDTH + (col - 1)] - 4 * E_prev_1D[index] + E_prev_1D[(row + 1) * WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);

		// PDE v2
		tempE[l_row][l_col] = tempE_prev[l_row][l_col] + alpha * (tempE_prev[l_row][l_col + 1] + tempE_prev[l_row][l_col - 1] - 4 * tempE_prev[l_row][l_col] + tempE_prev[l_row + 1][l_col] + tempE_prev[l_row - 1][l_col]);
		
		double e_temp = tempE[l_row][l_col];
		double r_temp = tempR[l_row][l_col];

		//ODE
		e_temp = e_temp - dt * (kk * e_temp * (e_temp - a) * (e_temp - 1) + e_temp * r_temp);
		r_temp = r_temp + dt * (epsilon + M1 * r_temp / (e_temp + M2)) * (-r_temp - kk * e_temp * (e_temp - b - 1));
		
		E_1D[index] = e_temp;
		R_1D[index] = r_temp;
	}

}

void simv1(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH, double *d_E_1D, double *d_E_prev_1D, double *d_R_1D)
{
	const dim3 block_size(TILE_DIM, TILE_DIM);
	const dim3 num_blocks(WIDTH / block_size.x, WIDTH / block_size.y);
	int Total_Bytes = WIDTH * WIDTH * sizeof(double);

	// ============ PDE Kernel ====
	cudaMemcpy(d_E_prev_1D, E_prev_1D, Total_Bytes, cudaMemcpyHostToDevice);

	simulate_version1_PDE<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);


	
	cudaMemcpy(E_1D, d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);

	// ============ ODE Kernel =======
	cudaMemcpy(d_E_1D, E_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_1D, R_1D, Total_Bytes, cudaMemcpyHostToDevice);

	simulate_version1_ODE<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);

	cudaMemcpy(E_1D, d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_1D, d_R_1D, Total_Bytes, cudaMemcpyDeviceToHost);
}

void simv2(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH, double *d_E_1D, double *d_E_prev_1D, double *d_R_1D)
{
	const dim3 block_size(TILE_DIM, TILE_DIM);
	const dim3 num_blocks(WIDTH / block_size.x, WIDTH / block_size.y);
	int Total_Bytes = WIDTH * WIDTH * sizeof(double);

	// Copy to GPU
	cudaMemcpy(d_E_prev_1D, E_prev_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_1D, E_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_1D, R_1D, Total_Bytes, cudaMemcpyHostToDevice);

	// Kernel Launch
	simulate_version2<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);

	// copy to Host
	cudaMemcpy(E_1D, d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_1D, d_R_1D, Total_Bytes, cudaMemcpyDeviceToHost);
}

void simv3(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH, double *d_E_1D, double *d_E_prev_1D, double *d_R_1D)
{
	const dim3 block_size(TILE_DIM, TILE_DIM);
	const dim3 num_blocks(WIDTH / block_size.x, WIDTH / block_size.y);
	int Total_Bytes = WIDTH * WIDTH * sizeof(double);

	// Copy to GPU
	cudaMemcpy(d_E_prev_1D, E_prev_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_1D, E_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_1D, R_1D, Total_Bytes, cudaMemcpyHostToDevice);

	// Kernel Launch
	simulate_version3<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);

	// copy to Host
	cudaMemcpy(E_1D, d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_1D, d_R_1D, Total_Bytes, cudaMemcpyDeviceToHost);
}

void simv4(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double *E_1D, double *E_prev_1D, double *R_1D, int WIDTH, double *d_E_1D, double *d_E_prev_1D, double *d_R_1D)
{
	const dim3 block_size(TILE_DIM, TILE_DIM);
	const dim3 num_blocks(WIDTH / block_size.x, WIDTH / block_size.y);
	int Total_Bytes = WIDTH * WIDTH * sizeof(double);

	// Copy to GPU
	cudaMemcpy(d_E_prev_1D, E_prev_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_1D, E_1D, Total_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R_1D, R_1D, Total_Bytes, cudaMemcpyHostToDevice);

	// Kernel Launch
	simulate_version4<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);

	// copy to Host
	cudaMemcpy(E_1D, d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_1D, d_R_1D, Total_Bytes, cudaMemcpyDeviceToHost);
}

// Main Refined -- Versioin 4 Refined --

// Main program
int main(int argc, char **argv)
{
	/*
	*  Solution arrays
	*   E is the "Excitation" variable, a voltage
	*   R is the "Recovery" variable
	*   E_prev is the Excitation variable for the previous timestep,
	*      and is used in time integration
	*/

	// cout<< "\n Hello I am waris"<<endl;
	int devId = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devId);
	printf("\nDevice : %s\n", prop.name);

	// For Serial Version
	double **E, **R, **E_prev;
	// For Host and GPU 
	double *E_1D, *R_1D, *E_prev_1D;
	double *d_E_1D, *d_E_prev_1D, *d_R_1D;

	// Various constants - these definitions shouldn't change
	const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

	double T = 1000.0;
	int m = 200, n = 200;
	int plot_freq = 0;
	int px = 1, py = 1;
	int no_comm = 0;
	int num_threads = 1;
	int WIDTH;

	cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
	m = n;
	// Allocate contiguous memory for solution arrays
	// The computational box is defined on [1:m+1,1:n+1]
	// We pad the arrays in order to facilitate differencing on the
	// boundaries of the computation box
	int Total_Bytes = (m + 2) * (n + 2) * sizeof(double);
	WIDTH = m + 2;

	E = alloc2D(m + 2, n + 2);
	E_prev = alloc2D(m + 2, n + 2);
	R = alloc2D(m + 2, n + 2);

	// Allocate space on the host (PINNED Memory)
	cudaError_t status = cudaMallocHost(&E_1D, Total_Bytes);
	status = cudaMallocHost(&E_prev_1D, Total_Bytes);
	status = cudaMallocHost(&R_1D, Total_Bytes);

	if (status != cudaSuccess) {
		printf("Error allocating pinned host memory\n");
	}
			
	// Allocate space on the GPU
	cudaMalloc(&d_E_1D, Total_Bytes);
	cudaMalloc(&d_E_prev_1D, Total_Bytes);
	cudaMalloc(&d_R_1D, Total_Bytes);

	
	int col, row;
	// Initialization
	for (row = 1; row <= m; row++)
	{
		for (col = 1; col <= n; col++)
		{
			E_prev[row][col] = 0;
			R[row][col] = 0;
			E_prev_1D[row * WIDTH + col] = 0;
			R_1D[row * WIDTH + col] = 0;
		}
	}

	for (row = 1; row <= m; row++)
	{
		for (col = n / 2 + 1; col <= n; col++)
		{
			E_prev[row][col] = 1.0;
			E_prev_1D[row * WIDTH + col] = 1.0;
		}
	}

	for (row = m / 2 + 1; row <= m; row++)
	{
		for (col = 1; col <= n; col++)
		{
			R[row][col] = 1.0;
			R_1D[row * WIDTH + col] = 1.0;
		}
	}

	
	double dx = 1.0 / n;

	// For time integration, these values shouldn't change
	double rp = kk * (b + 1) * (b + 1) / 4;
	double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
	double dtr = 1 / (epsilon + ((M1 / M2) * rp));
	double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
	double alpha = d * dt / (dx * dx);

	cout << "Grid Size       : " << n << endl;
	cout << "Duration of Sim : " << T << endl;
	cout << "Time step dt    : " << dt << endl;
	cout << "Process geometry: " << px << " x " << py << endl;

	if (no_comm)
	{
		cout << "Communication   : DISABLED" << endl;
	}

	cout << endl;

	// Start the timer
	double t0 = getTime();

	// Simulated time is different from the integer timestep number
	// Simulated time
	double t = 0.0;
	// Integer timestep number
	int niter = 0;
	
	int version = 4;

	while (t < T)
	{

		t += dt;
		niter++;

		mirrorBoundries(E_prev_1D, n, m, WIDTH);

		switch (version){

				// Size of the computational box
			case 1:
				simv1(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_1D, E_prev_1D, R_1D, 	WIDTH, d_E_1D, d_E_prev_1D, d_R_1D);
				break;
			case 2:
				simv2(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_1D, E_prev_1D, R_1D, 	WIDTH, d_E_1D, d_E_prev_1D, d_R_1D);
				break;
			case 3:
				simv3(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_1D, E_prev_1D, R_1D, 	WIDTH, d_E_1D, d_E_prev_1D, d_R_1D);
				break;
			case 4:
				simv4(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_1D, E_prev_1D, R_1D, 	WIDTH, d_E_1D, d_E_prev_1D, d_R_1D);
				break;
			default:
				cout<<"\nPlease Enter the Correct version"<<endl;
				return 0;
				
		}
		

		//swap current E with previous E
		double **tmp = E;
		E = E_prev;
		E_prev = tmp;

		double *tmp2 = E_1D;
		E_1D = E_prev_1D;
		E_prev_1D = tmp2;

	} //end of while loop

	double time_elapsed = getTime() - t0;

	double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
	double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

	cout << "Simulation Version          : " << version<<endl;
	cout << "Number of Iterations        : " << niter << endl;
	cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
	cout << "Sustained Gflops Rate       : " << Gflops << endl;
	cout << "Sustained Bandwidth (GB/sec): " << BW << endl<< endl;

	double mx;
	double l2norm = stats(E_prev, m, n, &mx);
	cout << "Max: " << mx << " L2norm: " << l2norm << endl;

	l2norm = stats1D(E_prev_1D, m, n, &mx, WIDTH);
	cout << "Max: " << mx << " L2norm: " << l2norm << " (1D or GPU)" <<endl;

	free(E);
	free(E_prev);
	free(R);

	cudaFreeHost(E_1D);
	cudaFreeHost(E_prev_1D);
	cudaFreeHost(R_1D);

	cudaFree(d_E_1D);
	cudaFree(d_E_prev_1D);
	cudaFree(d_R_1D);

	return 0;
}

//================================================== Utilities =========================================

// Mirror Ghost Boundries
void mirrorBoundries(double *E_prev_1D, const int n, const int m, const int WIDTH)
{
	/* 
	* Copy data from boundary of the computational box 
	* to the padding region, set up for differencing
	* on the boundary of the computational box
	* Using mirror boundaries
	*/

	int col, row;
	for (row = 1; row <= m; row++)
	{
		//E_prev[row][0] = E_prev[row][2];
		E_prev_1D[row * WIDTH + 0] = E_prev_1D[row * WIDTH + 2];
	}

	for (row = 1; row <= m; row++)
	{
		//E_prev[row][n + 1] = E_prev[row][n - 1];
		E_prev_1D[row * WIDTH + (n + 1)] = E_prev_1D[row * WIDTH + (n - 1)];
	}

	for (col = 1; col <= n; col++)
	{
		//E_prev[0][col] = E_prev[2][col];
		E_prev_1D[0 * WIDTH + col] = E_prev_1D[2 * WIDTH + col];
	}

	for (col = 1; col <= n; col++)
	{
		//E_prev[m + 1][col] = E_prev[m - 1][col];
		E_prev_1D[(m + 1) * WIDTH + col] = E_prev_1D[(m - 1) * WIDTH + col];
	}
}

// Allocate a 2D array
double **alloc2D(int m, int n)
{
	double **E;
	int nx = n, ny = m;
	E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
	assert(E);
	int row;
	for (row = 0; row < ny; row++)
		E[row] = (double *)(E + ny) + row * nx;
	return (E);
}

/* Reports statistics about the computation
	These values should not vary (except to within roundoff)
 when we use different numbers of  processes to solve the problem
 */
double stats(double **E, int m, int n, double *_mx)
{
	double mx = -1;
	double l2norm = 0;
	int col, row;
	for (row = 1; row <= m; row++)
		for (col = 1; col <= n; col++)
		{
			l2norm += E[row][col] * E[row][col];
			if (E[row][col] > mx)
				mx = E[row][col];
		}
	*_mx = mx;
	l2norm /= (double)((m) * (n));
	l2norm = sqrt(l2norm);
	return l2norm;
}

double stats1D(double *E, int m, int n, double *_mx, int WIDTH)
{
	double mx = -1;
	double l2norm = 0;
	int col, row;
	int index = -1;
	for (row = 1; row <= m; row++)
	{
		for (col = 1; col <= n; col++)
		{
			index = row * WIDTH + col;
			l2norm += E[index] * E[index];
			if (E[index] > mx)
			{
				mx = E[index];
			}
		}
	}

	*_mx = mx;
	l2norm /= (double)((m) * (n));
	l2norm = sqrt(l2norm);
	return l2norm;
}

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
	struct timeval TV;
	struct timezone TZ;

	const int RC = gettimeofday(&TV, &TZ);
	if (RC == -1)
	{
		cerr << "ERROR: Bad call to gettimeofday" << endl;
		return (-1);
	}

	return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));
}


void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads)
{
	/// Command line arguments
	// Default value of the domain sizes
	static struct option long_options[] = {
		{"n", required_argument, 0, 'n'},
		{"px", required_argument, 0, 'x'},
		{"py", required_argument, 0, 'y'},
		{"tfinal", required_argument, 0, 't'},
		{"plot", required_argument, 0, 'p'},
		{"nocomm", no_argument, 0, 'k'},
		{"numthreads", required_argument, 0, 'o'},
	};
	// Process command line arguments
	int ac;
	for (ac = 1; ac < argc; ac++)
	{
		int c;
		while ((c = getopt_long(argc, argv, "n:x:y:t:kp:o:", long_options, NULL)) != -1)
		{
			switch (c)
			{

				// Size of the computational box
			case 'n':
				n = atoi(optarg);
				break;

				// X processor geometry
			case 'x':
				px = atoi(optarg);

				// Y processor geometry
			case 'y':
				py = atoi(optarg);

				// Length of simulation, in simulated time units
			case 't':
				T = atof(optarg);
				break;
				// Turn off communication
			case 'k':
				no_comm = 1;
				break;

				// Plot the excitation variable
			case 'p':
				plot_freq = atoi(optarg);
				break;

				// Plot the excitation variable
			case 'o':
				num_threads = atoi(optarg);
				break;

				// Error
			default:
				printf("Usage: a.out [-n <domain size>] [-t <final time >]\n\t [-p <plot frequency>]\n\t[-px <x processor geometry> [-py <y proc. geometry] [-k turn off communication] [-o <Number of OpenMP threads>]\n");
				exit(-1);
			}
		}
	}
}