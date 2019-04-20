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

using namespace std;

// Utilities
//

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;

int my_rank, comm_size;
int my_chunk_width, my_chunk_height;
int no_of_vertical_strips, no_of_horizontal_strips;
int my_row, my_col;
int my_neighbors[4], my_neighbors_existance[4];
int my_abs_begin_r, my_abs_end_r, my_abs_begin_c, my_abs_end_c;

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
  int j;
  for (j = 0; j < ny; j++)
    E[j] = (double *)(E + ny) + j * nx;
  return (E);
}

// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
double stats(double **E, int m, int n, double *_mx)
{
  double mx = -1;
  double l2norm = 0;
  int i, j;
  for (j = 0; j < m; j++)
    for (i = 0; i < n; i++)
    {
      l2norm += E[j][i] * E[j][i];
      if (E[j][i] > mx)
        mx = E[j][i];
    }
  *_mx = mx;
  l2norm /= (double)((m) * (n));
  l2norm = sqrt(l2norm);
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
              const double M1, const double M2, const double b)
{
  int i, j;
  /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  if (my_neighbors_existance[3])
  {
    double exchange_with_left[m];
    //printf("waiting: %d", my_neighbors[3]);
    MPI_Recv(exchange_with_left, my_chunk_height, MPI_DOUBLE, my_neighbors[3], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (j = 1; j <= m; j++)
      E_prev[j][0] = exchange_with_left[j - 1];

    for (j = 1; j <= m; j++)
      exchange_with_left[j - 1] = E_prev[j][1];
    //printf("sending to: %d", my_neighbors[3]);
    MPI_Send(exchange_with_left, my_chunk_height, MPI_DOUBLE, my_neighbors[3], 0, MPI_COMM_WORLD);
  }
  else
  {
    for (j = 1; j <= m; j++)
      E_prev[j][0] = E_prev[j][2];
  }

  if (my_neighbors_existance[1])
  {
    double exchange_with_right[m];

    for (j = 1; j <= m; j++)
      exchange_with_right[j - 1] = E_prev[j][n];
    //printf("sending to: %d", my_neighbors[1]);
    MPI_Send(exchange_with_right, my_chunk_height, MPI_DOUBLE, my_neighbors[1], 0, MPI_COMM_WORLD);

    //printf("waiting: %d", my_neighbors[3]);
    MPI_Recv(exchange_with_right, my_chunk_height, MPI_DOUBLE, my_neighbors[1], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (j = 1; j <= m; j++)
      E_prev[j][n + 1] = exchange_with_right[j - 1];
  }
  else
  {
    for (j = 1; j <= m; j++)
      E_prev[j][n + 1] = E_prev[j][n - 1];
  }

  if (my_neighbors_existance[0])
  {
   // double exchange_with_top[m];
    //printf("waiting: %d", my_neighbors[0]);
    MPI_Recv(&E_prev[0][1], n, MPI_DOUBLE, my_neighbors[0], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //printf("sending to: %d", my_neighbors[0]);
    MPI_Send(&E_prev[1][1], n, MPI_DOUBLE, my_neighbors[0], 0, MPI_COMM_WORLD);
   /*  for (j = 1; j <= m; j++)
      E_prev[0][i] = exchange_with_top[j];

    for (j = 1; j <= m; j++)
      exchange_with_top[j] = E_prev[1][i]; */
  }
  else
  {
    for (i = 1; i <= n; i++)
      E_prev[0][i] = E_prev[2][i];
  }

  if (my_neighbors_existance[2])
  {
    //printf("sending to: %d", my_neighbors[2]);
    MPI_Send(&E_prev[m][1], n, MPI_DOUBLE, my_neighbors[2], 0, MPI_COMM_WORLD);
    //printf("waiting: %d", my_neighbors[2]);
    MPI_Recv(&E_prev[m + 1][1], n, MPI_DOUBLE, my_neighbors[2], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

/*     double exchange_with_bottom[m];
    for (j = 1; j <= m; j++)
      send_to_top[j] = E_prev[m][i];

    for (j = 1; j <= m; j++)
      E_prev[m][i] = send_to_top[j]; */
  }
  else
  {
    for (i = 1; i <= n; i++)
      E_prev[m + 1][i] = E_prev[m - 1][i];
  }

  // Solve for the excitation, the PDE
  for (j = 1; j <= m; j++)
  {
    for (i = 1; i <= n; i++)
    {
      E[j][i] = E_prev[j][i] + alpha * (E_prev[j][i + 1] + E_prev[j][i - 1] - 4 * E_prev[j][i] + E_prev[j + 1][i] + E_prev[j - 1][i]);
    }
  }

  /* 
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
  for (j = 1; j <= m; j++)
  {
    for (i = 1; i <= n; i++)
      E[j][i] = E[j][i] - dt * (kk * E[j][i] * (E[j][i] - a) * (E[j][i] - 1) + E[j][i] * R[j][i]);
  }

  for (j = 1; j <= m; j++)
  {
    for (i = 1; i <= n; i++)
      R[j][i] = R[j][i] + dt * (epsilon + M1 * R[j][i] / (E[j][i] + M2)) * (-R[j][i] - kk * E[j][i] * (E[j][i] - b - 1));
  }
}

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
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  double **E, **E_prev;
  double **my_E, **my_R, **my_E_prev;
  // Various constants - these definitions shouldn't change
  const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

  double T;
  int m, n;
  int plot_freq;
  int px, py;
  int no_comm;
  int num_threads;
  int parameters_to_broadcast[9];
  //printf("my rank: %d\n", my_rank);
  if (my_rank == 0)
  {
    T = 1000.0;
    m = 8, n = 8;
    plot_freq = 0;
    px = 1, py = 1;
    no_comm = 0;
    num_threads = 1;
    cmdLine(argc, argv, T, n, px, py, plot_freq, no_comm, num_threads);
    m = n;
    no_of_vertical_strips = 2;
    no_of_horizontal_strips = 1;

    parameters_to_broadcast[0] = m;
    parameters_to_broadcast[1] = n;
    parameters_to_broadcast[2] = plot_freq;
    parameters_to_broadcast[3] = px;
    parameters_to_broadcast[4] = py;
    parameters_to_broadcast[5] = no_comm;
    parameters_to_broadcast[6] = num_threads;
    parameters_to_broadcast[7] = no_of_horizontal_strips;
    parameters_to_broadcast[8] = no_of_vertical_strips;
  }
  MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&parameters_to_broadcast, 9, MPI_INT, 0, MPI_COMM_WORLD);
  
  m = parameters_to_broadcast[0];
  n = parameters_to_broadcast[1];
  plot_freq = parameters_to_broadcast[2];
  px = parameters_to_broadcast[3];
  py = parameters_to_broadcast[4];
  no_comm = parameters_to_broadcast[5];
  num_threads = parameters_to_broadcast[6];
  no_of_horizontal_strips = parameters_to_broadcast[7];
  no_of_vertical_strips = parameters_to_broadcast[8];

  E_prev = alloc2D(m, n);

  my_chunk_width = n / no_of_vertical_strips;
  my_chunk_height = m / no_of_horizontal_strips;

  my_row = my_rank / no_of_vertical_strips;
  my_col = my_rank % no_of_vertical_strips;

  my_abs_begin_r = my_row * my_chunk_height;
  my_abs_end_r = my_abs_begin_r + my_chunk_height;
  my_abs_begin_c = my_col * my_chunk_width;
  my_abs_end_c = my_abs_begin_c + my_chunk_width;

  my_neighbors_existance[0] = my_row - 1 < 0 ? 0 : 1;
  my_neighbors_existance[1] = my_col + 1 >= no_of_vertical_strips ? 0 : 1;
  my_neighbors_existance[2] = my_row + 1 >= no_of_horizontal_strips ? 0 : 1;
  my_neighbors_existance[3] = my_col - 1 < 0 ? 0 : 1;

  my_neighbors[0] = (my_row - 1) * no_of_vertical_strips + my_col;
  my_neighbors[1] = my_row * no_of_vertical_strips + my_col + 1;
  my_neighbors[2] = (my_row + 1) * no_of_vertical_strips + my_col;
  my_neighbors[3] = my_row * no_of_vertical_strips + my_col - 1;

  my_E = alloc2D(my_chunk_height + 2, my_chunk_width + 2);
  my_R = alloc2D(my_chunk_height + 2, my_chunk_width + 2);
  my_E_prev = alloc2D(my_chunk_height + 2, my_chunk_width + 2);

  int ones_begin_c = max(my_abs_begin_c, n / 2);
 // int ones_end_c = my_abs_end_c - ones_begin_c;
  ones_begin_c = ones_begin_c - my_abs_begin_c + 1;
  int ones_begin_r = max(my_abs_begin_r, m / 2);
 // int ones_end_r = my_abs_end_r - ones_begin_r;
  ones_begin_r = ones_begin_r - my_abs_begin_r + 1;

  int i, j;
  // Initialization
  for (j = 1; j <= my_chunk_height; j++)
    for (i = 1; i <= my_chunk_width; i++)
      my_E_prev[j][i] = my_R[j][i] = 0;

  for (j = 1; j <= my_chunk_height; j++)
    for (i = ones_begin_c; i <= my_chunk_width; i++)
      my_E_prev[j][i] = 1.0;

  for (j = ones_begin_r; j <= my_chunk_height; j++)
    for (i = 1; i <= my_chunk_width; i++)
      my_R[j][i] = 1.0;


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

  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter = 0;
  while (t < T)
  {

    t += dt;
    niter++;

    simulate(my_E, my_E_prev, my_R, alpha, my_chunk_width, my_chunk_height, kk, dt, a, epsilon, M1, M2, b);

    //swap current E with previous E
    double **tmp = my_E;
    my_E = my_E_prev;
    my_E_prev = tmp;

    if (plot_freq)
    {
      int k = (int)(t / plot_freq);
      if ((t - k * plot_freq) < dt)
      {
        splot(E, t, niter, m + 2, n + 2);
      }
    }
  } //end of while loop

  double ** E_prev_to_send = alloc2D(my_chunk_height, my_chunk_width);
for(int r = 0; r < my_chunk_height; r++)
    memcpy(&E_prev_to_send[r][0], &my_E_prev[r + 1][1], sizeof(double) * my_chunk_width);

  MPI_Gather(&E_prev_to_send[0][0], my_chunk_width * my_chunk_height, MPI_DOUBLE, &E_prev[0][0], my_chunk_height * my_chunk_width, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

    double mx;
    double l2norm = stats(E_prev, m, n, &mx);

    cout << "Max: " << mx << " L2norm: " << l2norm << endl;

    if (plot_freq)
    {
      cout << "\n\nEnter any input to close the program and the plot..." << endl;
      getchar();
    }
  }
  free(E_prev);

  MPI_Finalize();

  return 0;
}
