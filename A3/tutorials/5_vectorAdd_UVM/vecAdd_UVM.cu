/*

	Implement your CUDA kernel in this file

*/



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


// External functions
//extern "C" void splot(double **E, double T, int niter, int m, int n);

void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads){
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

// Utilities
//

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;

double getTime(){
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

// end getTime()

// Allocate a 2D array
double **alloc2D(int m, int n){
	double **E;
	int nx = n, ny = m;
	E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
	assert(E);
	int row;
	for (row = 0; row < ny; row++)
		E[row] = (double *)(E + ny) + row * nx;
	return (E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx){
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

double stats1D(double *E, int m, int n, double *_mx, int WIDTH){
	double mx = -1;
	double l2norm = 0;
	int col, row;
	int index = -1;
	for (row = 1; row <= m; row++) {
		for (col = 1; col <= n; col++){
			index = row * WIDTH + col;
			l2norm += E[index] * E[index];
			if (E[index] > mx) {
				mx = E[index];
			}
		}
	}
		
	*_mx = mx;
	l2norm /= (double)((m) * (n));
	l2norm = sqrt(l2norm);
	return l2norm;
}

__global__ void simulate_version1(const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double* E_1D, double* E_prev_1D, double* R_1D, int WIDTH){

   
  int RADIUS = 1;
  int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
  int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
  
  if (row < (WIDTH -1) && col < (WIDTH -1)) {
      E_1D[row*WIDTH+col] = E_prev_1D[row*WIDTH+col] + alpha * (E_prev_1D[row*WIDTH+(col+1)] + E_prev_1D[row*WIDTH + (col-1)] - 4 * E_prev_1D[row*WIDTH+ col] + E_prev_1D[(row + 1)*WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);
  }   

}

void simulate(double **E, double **E_prev, double **R, const double alpha, const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, double* E_1D, double* E_prev_1D, double* R_1D, int WIDTH){
	
	int col, row;
	/* 
	* Copy data from boundary of the computational box 
	* to the padding region, set up for differencing
	* on the boundary of the computational box
	* Using mirror boundaries
	*/

	for (row = 1; row <= m; row++) {
		E_prev[row][0] = E_prev[row][2];
		E_prev_1D[row* WIDTH + 0] = E_prev_1D[row*WIDTH + 2];
	}
	
	for (row = 1; row <= m; row++) {
		E_prev[row][n + 1] = E_prev[row][n - 1];
		E_prev_1D[row*WIDTH+ (n+1)] = E_prev_1D[row*WIDTH + (n-1)];
	}

	for (col = 1; col <= n; col++) {
		E_prev[0][col] = E_prev[2][col];
		E_prev_1D[0*WIDTH+col] = E_prev_1D[2*WIDTH+ col];
	}

	for (col = 1; col <= n; col++) {
		E_prev[m + 1][col] = E_prev[m - 1][col];
		E_prev_1D[(m+1)*WIDTH+ col] = E_prev_1D[(m-1)*WIDTH+ col];
    }
    
    // ================ Kernel Setup ==============
    // double *d_E_1D, *d_E_prev_1D, *d_R_1D;
    // const dim3 block_size(TILE_DIM, TILE_DIM);
    // const dim3 num_blocks(WIDTH / block_size.x, WIDTH / block_size.y);
    // int Total_Bytes = WIDTH * WIDTH * sizeof(double);

    // // Allocate space on the device
    // cudaMalloc( &d_E_1D, Total_Bytes) ;
    // cudaMalloc( &d_E_prev_1D,Total_Bytes) ;
    // cudaMalloc(&d_R_1D, Total_Bytes);
    
    // // Copy input data to device
    // cudaMemcpy(d_E_1D, E_1D, Total_Bytes, cudaMemcpyHostToDevice) ;
    // cudaMemcpy(d_E_prev_1D, E_prev_1D, Total_Bytes, cudaMemcpyHostToDevice) ;
    // cudaMemcpy(d_R_1D, R_1D, Total_Bytes, cudaMemcpyHostToDevice) ;
    
    // // simulate_version1<<<num_blocks, block_size>>>(alpha, n, m, kk, dt, a, epsilon, M1, M2, b, d_E_1D, d_E_prev_1D, d_R_1D, WIDTH);

    // cudaMemcpy(E_1D ,d_E_1D, Total_Bytes, cudaMemcpyDeviceToHost);

    // cudaFree(d_E_1D);
    // cudaFree(d_E_prev_1D);
    // cudaFree(d_R_1D);
    // =================================================

	// Solve for the excitation, the PDE
	for (row = 1; row <= m; row++){
		for (col = 1; col <= n; col++){
            E[row][col] = E_prev[row][col] + alpha * (E_prev[row][col + 1] + E_prev[row][col - 1] - 4 * E_prev[row][col] + E_prev[row + 1][col] + E_prev[row - 1][col]);
            
			E_1D[row*WIDTH+col] = E_prev_1D[row*WIDTH+col] + alpha * (E_prev_1D[row*WIDTH+(col+1)] + E_prev_1D[row*WIDTH + (col-1)] - 4 * E_prev_1D[row*WIDTH+ col] + E_prev_1D[(row + 1)*WIDTH + col] + E_prev_1D[(row - 1) * WIDTH + col]);
		}
	}

	/* 
	* Solve the ODE, advancing excitation and recovery to the
	*     next timtestep
	*/
	int index = -1;
	for (row = 1; row <= m; row++){
		for (col = 1; col <= n; col++) {
			E[row][col] = E[row][col] - dt * (kk * E[row][col] * (E[row][col] - a) * (E[row][col] - 1) + E[row][col] * R[row][col]);
			index = row * WIDTH + col;	
			E_1D[index] = E_1D[index] - dt * (kk * E_1D[index] * (E_1D[index] - a) * (E_1D[index] - 1) + E_1D[index] * R_1D[index]);

		}
	}

	for (row = 1; row <= m; row++){
		for (col = 1; col <= n; col++) {
			
			R[row][col] = R[row][col] + dt * (epsilon + M1 * R[row][col] / (E[row][col] + M2)) * (-R[row][col] - kk * E[row][col] * (E[row][col] - b - 1));

			index =  row * WIDTH + col;
			R_1D[index] = R_1D[index] + dt * (epsilon + M1 * R_1D[index] / (E_1D[index] + M2)) * (-R_1D[index] - kk * E_1D[index] * (E_1D[index] - b - 1));
		}
	}

}

// Main program
int main(int argc, char **argv){
	/*
	*  Solution arrays
	*   E is the "Excitation" variable, a voltage
	*   R is the "Recovery" variable
	*   E_prev is the Excitation variable for the previous timestep,
	*      and is used in time integration
	*/
    cout<< "\n Hello I am waris"<<endl;
	double **E, **R, **E_prev;

	double * E_1D, *R_1D, *E_prev_1D; 

	// Various constants - these definitions shouldn't change
	const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

	double T = 1000.0;
	int m = 200, n = 200;
	int plot_freq = 0;
	int px = 1, py = 1;
	int no_comm = 0;
	int num_threads = 1;
	int WIDTH;

	//int size;

	cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
	m = n;
	// Allocate contiguous memory for solution arrays
	// The computational box is defined on [1:m+1,1:n+1]
	// We pad the arrays in order to facilitate differencing on the
	// boundaries of the computation box
	int Total_Bytes = (m+2) * (n+2) *sizeof(double);
	WIDTH =  m+2;


	E = alloc2D(m + 2, n + 2);
	E_prev = alloc2D(m + 2, n + 2);
	R = alloc2D(m + 2, n + 2);
	
	E_1D = (double *) malloc (Total_Bytes);
	E_prev_1D = (double *) malloc (Total_Bytes);
    R_1D = (double *) malloc (Total_Bytes);
    
    


	int col, row;
	// Initialization
	
	for (row = 1; row <= m; row++) {
		for (col = 1; col <= n; col++) {
			E_prev[row][col] = 0;
			R[row][col] = 0;
			E_prev_1D[row*WIDTH + col] = 0;
			R_1D[row*WIDTH + col] = 0;
		}
	}
		
	for (row = 1; row <= m; row++) {
		for (col = n / 2 + 1; col <= n; col++) {
			E_prev[row][col] = 1.0;
			E_prev_1D[row*WIDTH + col] = 1.0;
		}
	}
		
	for (row = m / 2 + 1; row <= m; row++) {
		for (col = 1; col <= n; col++) {
			R[row][col] = 1.0;
			R_1D[row*WIDTH + col] = 1.0;
		}
	}

    //cout<< "check 1"<<endl;
		

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
	
	if (no_comm) {
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

	while (t < T){

		t += dt;
		niter++;

		simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, E_1D, E_prev_1D, R_1D, WIDTH);

		//swap current E with previous E
		double **tmp = E;
		E = E_prev;
		E_prev = tmp;

		double *tmp2 = E_1D;
		E_1D = E_prev_1D;
		E_prev_1D = tmp2;


		// if (plot_freq){
		// 	int k = (int)(t / plot_freq);
		// 	if ((t - k * plot_freq) < dt){
		// 		splot(E, t, niter, m + 2, n + 2);
		// 	}
		// }

	} //end of while loop

	double time_elapsed = getTime() - t0;

	double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
	double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

	cout << "Number of Iterations        : " << niter << endl;
	cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
	cout << "Sustained Gflops Rate       : " << Gflops << endl;
	cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;

	double mx;
	double l2norm = stats(E_prev, m, n, &mx);
	cout << "Max: " << mx << " L2norm: " << l2norm << endl;

	l2norm = stats1D(E_prev_1D, m, n, &mx, WIDTH);
	cout << "1D Max : " << mx << " 1D L2norm: " << l2norm << endl;

	// if (plot_freq){
	// 	cout << "\n\nEnter any input to close the program and the plot..." << endl;
	// 	getchar();
	// }

	free(E);
	free(E_prev);
	free(R);

	free(E_1D);
	free(E_prev_1D);
	free(R_1D);

	return 0;
}


/* **********************************************************
*  Author : Urvashi R.V. [04/06/2004]
*      Modified by Didem Unat [03/23/18]
*************************************************************/

//#include <stdio.h>

/* Function to plot the 2D array
* 'gnuplot' is instantiated via a pipe and 
* the values to be plotted are passed through, along 
* with gnuplot commands */

// FILE *gnu = NULL;

// void splot(double **U, double T, int niter, int m, int n){
// 	int col, row;
// 	if (gnu == NULL)
// 		gnu = popen("gnuplot", "w");

// 	double mx = -1, mn = 32768;
// 	for (row = 0; row < m; row++)
// 		for (col = 0; col < n; col++)
// 		{
// 			if (U[row][col] > mx)
// 				mx = U[row][col];
// 			if (U[row][col] < mn)
// 				mn = U[row][col];
// 		}

// 	fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
// 	fprintf(gnu, "set size square\n");
// 	fprintf(gnu, "set key off\n");
// 	fprintf(gnu, "set pm3d map\n");
// 	// Various color schemes
// 	fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");

// 	//    fprintf(gnu,"set palette rgbformulae 22, 13, 31\n");
// 	//    fprintf(gnu,"set palette rgbformulae 30, 31, 32\n");

// 	fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);
// 	for (row = 0; row < m; row++)
// 	{
// 		for (col = 0; col < n; col++)
// 		{
// 			fprintf(gnu, "%d %d %f\n", col, row, U[col][row]);
// 		}
// 		fprintf(gnu, "\n");
// 	}
// 	fprintf(gnu, "e\n");
// 	fflush(gnu);
// 	return;
// }