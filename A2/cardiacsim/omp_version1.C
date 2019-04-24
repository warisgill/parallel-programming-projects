/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
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
#include <mpi.h>
#include "omp.h"
using namespace std;

// Utilities
//

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

} // end getTime()

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

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
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
    //l2norm /= (double)((m) * (n));
    //l2norm = sqrt(l2norm);
    return l2norm;
}

// External functions
extern "C"
{
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &no_comm, int &num_threads);

void simulate(double **E, double **E_prev, double **R,
              const double alpha, const int n, const int m, const double kk,
              const double dt, const double a, const double epsilon,
              const double M1, const double M2, const double b, int my_rank, int comm_sz, int up, int down, int right, int left, double **buffer, int no_comm, int thread_nums)
{
    int col, row;
    /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */

    #pragma omp single 
    {
        int tag1 = 0;
        int tag2 = 1;
        int i = 0;
        MPI_Status stat;

        MPI_Request req1;
        MPI_Request req2;
        MPI_Request req3;
        MPI_Request req4;

        MPI_Request req5;
        MPI_Request req6;
        MPI_Request req7;
        MPI_Request req8;
    }
    

    // === Sending Data ===
    # pragma omp single
    if (no_comm == 0)
    {
        if (up > -1)
        {
            MPI_Isend(&E_prev[1][1], n, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req1);
        }

        if (down > -1)
        {
            MPI_Isend(&E_prev[m][1], n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req2);
        }

        if (left > -1)
        {
            //#pragma omp parallel for num_threads(thread_nums)
            for (i = 1; i <= m; i++)
            {
                buffer[0][i - 1] = E_prev[i][1];
            }

            MPI_Isend(&buffer[0][0], m, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &req5);
        }

        if (right > -1)
        {   
            //#pragma omp parallel for num_threads(thread_nums)
            for (i = 1; i <= m; i++)
            {
                buffer[1][i - 1] = E_prev[i][n];
            }

            MPI_Isend(&buffer[1][0], m, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &req6);
        }
    }

    //#pragma omp parallel num_threads(thread_nums)
    //{
        #pragma omp for schedule(static)
        for (row = 1; row <= m; row++){
            E_prev[row][0] = E_prev[row][2];
        }
        
        #pragma omp for schedule(static)
        for (row = 1; row <= m; row++){
            E_prev[row][n + 1] = E_prev[row][n - 1];
        }

        #pragma omp for schedule(static)
        for (col = 1; col <= n; col++){
            E_prev[0][col] = E_prev[2][col];
        }

        #pragma omp for schedule(static)
        for (col = 1; col <= n; col++){
            E_prev[m + 1][col] = E_prev[m - 1][col];
        }
    //}
    

    // ===== Recving Data ====
    #pragma omp single
    if (no_comm == 0)
    {
        if (down > -1)
        {
            MPI_Irecv(&E_prev[m + 1][1], n, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req3);
        }

        if (up > -1)
        {
            MPI_Irecv(&E_prev[0][1], n, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req4);
        }

        if (left > -1)
        {
            MPI_Irecv(&buffer[2][0], m, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &req7);
        }

        if (right > -1)
        {
            MPI_Irecv(&buffer[3][0], m, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &req8);
        }

        // ====Wait on Recv Data ===

        if (down > -1)
        {
            MPI_Wait(&req3, &stat);
        }

        if (up > -1)
        {
            MPI_Wait(&req4, &stat);
        }

        if (left > -1)
        {

            MPI_Wait(&req7, &stat);
            //#pragma omp parallel for num_threads(thread_nums)
            for (i = 1; i <= m; i++)
            {
                E_prev[i][0] = buffer[2][i - 1];
            }
        }

        if (right > -1)
        {
            MPI_Wait(&req8, &stat);
            //#pragma omp parallel for num_threads(thread_nums)
            for (i = 1; i <= m; i++)
            {
                E_prev[i][n + 1] = buffer[3][i - 1];
            }
        }
    }

    //===== Solve for the excitation, the PDE ====
    // #pragma omp parallel num_threads(thread_nums)
    // {
        #pragma omp for collapse(2) private(row,col) schedule(static)
        for (row = 1; row <= m; row++)
        {
            for (col = 1; col <= n; col++)
            {
                E[row][col] = E_prev[row][col] + alpha * (E_prev[row][col + 1] + E_prev[row][col - 1] - 4 * E_prev[row][col] + E_prev[row + 1][col] + E_prev[row - 1][col]);
            }
        }

        /* 
        * Solve the ODE, advancing excitation and recovery to the
        *     next timtestep
        */
       #pragma omp for collapse(2) private(row,col) schedule(static)
        for (row = 1; row <= m; row++)
        {
            for (col = 1; col <= n; col++)
                E[row][col] = E[row][col] - dt * (kk * E[row][col] * (E[row][col] - a) * (E[row][col] - 1) + E[row][col] * R[row][col]);
        }

        #pragma omp for collapse(2) private(row,col) schedule(static)
        for (row = 1; row <= m; row++)
        {
            for (col = 1; col <= n; col++)
                R[row][col] = R[row][col] + dt * (epsilon + M1 * R[row][col] / (E[row][col] + M2)) * (-R[row][col] - kk * E[row][col] * (E[row][col] - b - 1));
        }
    //}
        

    // Wait on sent data
    #pragma omp single
    if (no_comm == 0)
    {
        if (up > -1)
        {
            MPI_Wait(&req1, &stat);
        }

        if (down > -1)
        {
            MPI_Wait(&req2, &stat);
        }

        if (left > -1)
        {
            MPI_Wait(&req5, &stat);
        }

        if (right > -1)
        {
            MPI_Wait(&req6, &stat);
        }
    }
}

void initializeArrays(double **E, double **E_prev, double **R, int m, int n);
void setNeighbours(int my_rank, int *up, int *down, int *left, int *right, int *my_i, int *my_j, int px, int py);
void print2DArray(double **arr, int m, int n);

// ================================================== Main program ========================================================================

int main(int argc, char **argv)
{
    double **E, **R, **E_prev;
    // Various constants - these definitions shouldn't change
    const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

    double T = 1000.0;
    int m = 400, n = 400;
    int plot_freq = 0;
    int px = 1, py = 1;
    int no_comm = 0;
    int num_threads = 1;

    double l2norm;
    double mx;
    double **right_left_send_recv_buffer;

    // ==== Local Variables ====
    int my_rank, comm_sz;
    double **E_local, **R_local, **E_prev_local;
    int m_local;
    int n_local;
    int right;
    int left;
    int up;
    int down;
    int i, j;
    
    //=====Initializing MPI====
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Request req1;
    MPI_Request req2;
    MPI_Status stat;

    cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
    //num_threads = 5;
    m = n;
    int my_i = -1;
    int my_j = -1;
    setNeighbours(my_rank, &up, &down, &left, &right, &my_i, &my_j, px, py);

    m_local = m / py; // rows
    n_local = n / px; //columns

    // =======================  Handling Corner Cases ==============
    int x_axis_array[px];
    int y_axis_array[py];

    //rows
    for (i = 0; i < py; i++)
    {
        y_axis_array[i] = m_local;
    }

    for (i = 0; i < m % py; i++)
    { //remainder
        y_axis_array[i] += 1;
    }

    for (i = 1; i < py; i++)
    {
        y_axis_array[i] = y_axis_array[i] + y_axis_array[i - 1];
    }

    //cols
    for (i = 0; i < px; i++)
    {
        x_axis_array[i] = n_local;
    }
    for (i = 0; i < n % px; i++)
    { //remainder
        x_axis_array[i] += 1;
    }

    for (i = 1; i < px; i++)
    {
        x_axis_array[i] = x_axis_array[i] + x_axis_array[i - 1];
    }

    int start_col = 1;
    int end_col = x_axis_array[my_j - 1] + 1;

    int start_row = 1;
    int end_row = y_axis_array[my_i - 1] + 1;

    if ((my_i - 2) > -1)
    {
        start_row = y_axis_array[my_i - 2] + 1;
    }

    if ((my_j - 2) > -1)
    {
        start_col = x_axis_array[my_j - 2] + 1;
    }

    // updated with corner case
    m_local = end_row - start_row;
    n_local = end_col - start_col;

    right_left_send_recv_buffer = alloc2D(4, m_local);
    E_local = alloc2D(m_local + 2, n_local + 2);
    E_prev_local = alloc2D(m_local + 2, n_local + 2);
    R_local = alloc2D(m_local + 2, n_local + 2);
    E = alloc2D(m + 2, n + 2);
    E_prev = alloc2D(m + 2, n + 2);
    R = alloc2D(m + 2, n + 2);

    initializeArrays(E, E_prev, R, m, n);
    int temp_i = 1, temp_j = 1;
    for (i = start_row; i < end_row; i++)
    {
        temp_j = 1;
        for (j = start_col; j < end_col; j++)
        {
            E_prev_local[temp_i][temp_j] = E_prev[i][j];
            R_local[temp_i][temp_j] = R[i][j];
            temp_j++;
        }
        temp_i++;
    }

    free(E);
    free(E_prev);
    free(R);
    //==================================

    double dx = 1.0 / n;

    // For time integration, these values shouldn't change
    double rp = kk * (b + 1) * (b + 1) / 4;
    double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
    double dtr = 1 / (epsilon + ((M1 / M2) * rp));
    double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
    double alpha = d * dt / (dx * dx);

    if (my_rank == 0)
    {
        cout << "Grid Size       : " << n << endl;
        cout << "Duration of Sim : " << T << endl;
        cout << "Time step dt    : " << dt << endl;
        cout << "Process geometry: " << px << " x " << py << endl;
        if (no_comm)
            cout << "Communication   : DISABLED" << endl;

        cout << endl;
    }

    // Start the timer
    double t0 = getTime();

    double t = 0.0;
    int niter = 0;
    #pragma omp parallel
    while (t < T)
    {
        #pragma omp single
        {
            t += dt;
            niter++;
        }
        


        simulate(E_local, E_prev_local, R_local, alpha, n_local, m_local, kk, dt, a, epsilon, M1, M2, b, my_rank, comm_sz, up, down, right, left, right_left_send_recv_buffer, no_comm,num_threads);

        //swap current E with previous E
        #pragma omp single
        {
            double **tmp = E_local;
            E_local = E_prev_local;
            E_prev_local = tmp;

            if (plot_freq)
            {
                int k = (int)(t / plot_freq);
                if ((t - k * plot_freq) < dt)
                {
                    splot(E_local, t, niter, m + 2, n + 2);
                }
            }
        }
        
    } //end of while loop

    double mx_local;
    double l2norm_local = stats(E_prev_local, m_local, n_local, &mx_local);

    // MPI_Reduce(&mx_local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Reduce(&l2norm_local, &l2norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (no_comm == 0)
    {
        MPI_Ireduce(&mx_local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD, &req1);
        MPI_Ireduce(&l2norm_local, &l2norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req2);
        MPI_Wait(&req1, &stat);
        MPI_Wait(&req2, &stat);
    }

    double time_elapsed = getTime() - t0;

    if (my_rank == 0)
    {
        double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
        double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

        cout << "Number of Iterations        : " << niter << endl;
        cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
        cout << "Sustained Gflops Rate       : " << Gflops << endl;
        cout << "Sustained Bandwidth (GB/sec): " << BW << endl
             << endl;

        l2norm /= (double)((m) * (n));
        l2norm = sqrt(l2norm);
        cout << "Max: " << mx << " L2norm: " << l2norm << endl;

        if (plot_freq)
        {
            cout << "\n\nEnter any input to close the program and the plot..." << endl;
            getchar();
        }
    }

    free(E_local);
    free(E_prev_local);
    free(R_local);
    free(right_left_send_recv_buffer);

    MPI_Finalize();
    return 0;
}

void setNeighbours(int my_rank, int *up, int *down, int *left, int *right, int *my_i, int *my_j, int px, int py)
{
    int g[py + 2][px + 2]; // processes geometery
    int process_id = 0;
    int i, j;
    for (i = 0; i < py + 2; i++)
    {
        for (j = 0; j < px + 2; j++)
        {
            g[i][j] = -1;
        }
    }

    for (i = 1; i <= py; i++)
    {
        for (j = 1; j <= px; j++)
        {

            if (process_id == my_rank)
            {
                *my_i = i;
                *my_j = j;
            }
            g[i][j] = process_id;
            process_id++;
        }
    }

    *up = g[*my_i - 1][*my_j];
    *down = g[*my_i + 1][*my_j];
    *left = g[*my_i][*my_j - 1];
    *right = g[*my_i][*my_j + 1];
}

void print2DArray(double **arr, int m, int n)
{

    int row, col;
    for (row = 0; row < m + 2; row++)
    {
        for (col = 0; col < n + 2; col++)
        {
            cout << arr[row][col] << " ";
        }
        cout << endl;
    }
}

void initializeArrays(double **E, double **E_prev, double **R, int m, int n)
{
    int col, row;
    // Initialization
    for (row = 1; row <= m; row++)
        for (col = 1; col <= n; col++)
            E_prev[row][col] = R[row][col] = 0;

    for (row = 1; row <= m; row++)
        for (col = n / 2 + 1; col <= n; col++)
            E_prev[row][col] = 1.0;

    for (row = m / 2 + 1; row <= m; row++)
        for (col = 1; col <= n; col++)
            R[row][col] = 1.0;
}

/*
    n = 400;
    m = 400;
    Max: 0.964989 L2norm: 0.652593
*/