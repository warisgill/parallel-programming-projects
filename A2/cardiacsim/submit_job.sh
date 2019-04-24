#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=cardiac-sim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --time=30:00
#SBATCH --output=partd1-again%j.out
#SBATCH --mem-per-cpu=1000M

################################################################################
## Load openmpi version 3.0.0
echo "Loading openmpi module ..."
module load openmpi/3.0.0

## Load GCC-7.3.0
echo "Loading GCC module ..."
module load gcc/7.3.0



# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo


# Different MPI configurations
echo "With OPENMP"

echo "2 MPI and 16 Threads 2X1"
export OMP_NUM_THREADS=16
mpirun -np 2 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 16 -x 2 -y 1

echo "2 MPI 2X1"
mpirun -np 2 ./d2_cardiacsim -n 1024 -t 100 -x 2 -y 1


echo "2 MPI and 16 Threads 1X2"
export OMP_NUM_THREADS=16
mpirun -np 2 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 16 -x 1 -y 2

echo "2 MPI 1X2"
mpirun -np 2 ./d2_cardiacsim -n 1024 -t 100 -x 1 -y 2


echo "4 MPI  and 8 Threads 4X1"
export OMP_NUM_THREADS=8
mpirun -np 4 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 8 -x 4 -y 1
echo "4 MPI 4X1"
mpirun -np 4 ./d2_cardiacsim -n 1024 -t 100 -x 4 -y 1


echo "4 MPI  and 8 Threads 1X4"
export OMP_NUM_THREADS=8
mpirun -np 4 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 8 -x 1 -y 4
echo "4 MPI 1X4"
mpirun -np 4 ./d2_cardiacsim -n 1024 -t 100 -x 1 -y 4

echo "8 MPI and 4 Threads 8X1"
export OMP_NUM_THREADS=4
mpirun -np 8 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 4 -x 8 -y 1

echo "8 MPI 8X1"
mpirun -np 8 ./d2_cardiacsim -n 1024 -t 100 -x 8 -y 1


echo "8 MPI and 4 Threads 1X8"
export OMP_NUM_THREADS=4
mpirun -np 8 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 4 -x 1 -y 8
echo "8 MPI 1X8"
mpirun -np 8 ./d2_cardiacsim -n 1024 -t 100 -x 1 -y 8

echo "16 MPI and 2 Threads 16X1"
export OMP_NUM_THREADS=2
mpirun -np 16 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 2 -x 16 -y 1

echo "16 MPI 16X1"
mpirun -np 16 ./d2_cardiacsim -n 1024 -t 100 -x 16 -y 1


echo "16 MPI and 2 Threads 1X16"
export OMP_NUM_THREADS=2
mpirun -np 16 -bind-to socket -map-by socket ./openmp_d2_cardiacsim -n 1024 -t 100 -o 2 -x 1 -y 16

echo "16 MPI 1X16"
mpirun -np 16 ./d2_cardiacsim -n 1024 -t 100 -x 1 -y 16

#....
echo "Finished with execution!"