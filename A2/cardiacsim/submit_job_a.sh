#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=cardiac-sim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=short
#SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=30:00
#SBATCH --output=test-cardiacsim-a-%j.out

################################################################################
################################################################################

## Load openmpi version 3.0.0
echo "Loading openmpi module ..."
module load openmpi/3.0.0

## Load GCC-7.3.0
echo "Loading GCC module ..."
module load gcc/7.3.0

echo ""
echo "======================================================================================"

echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Serial version ..."
./cardiacsim -n 1024 -t 100

# Different MPI configurations
echo "Without OPENMP"
echo "1 MPI"
mpirun -np 1 ./cardiacsim2D -n 1024 -t 100

echo "2 MPI 2X1"
mpirun -np 2 ./cardiacsim2D -n 1024 -t 100 -x 2 -y 1

echo "2 MPI 1X2"
mpirun -np 2 ./cardiacsim2D -n 1024 -t 100 -x 1 -y 2

echo "4 MPI 1X4"
mpirun -np 4 ./cardiacsim2D -n 1024 -t 100 -x 1 -y 4

echo "4 MPI 2X2"
mpirun -np 4 ./cardiacsim2D -n 1024 -t 100 -x 2 -y 2

echo "4 MPI 4X1"
mpirun -np 4 ./cardiacsim2D -n 1024 -t 100 -x 4 -y 1

echo "8 MPI 1X8"
mpirun -np 8 ./cardiacsim2D -n 1024 -t 100 -x 1 -y 8

echo "8 MPI 2X4"
mpirun -np 8 ./cardiacsim2D -n 1024 -t 100 -x 2 -y 4

echo "8 MPI 4X2"
mpirun -np 8 ./cardiacsim2D -n 1024 -t 100 -x 4 -y 2

echo "8 MPI 8X1"
mpirun -np 8 ./cardiacsim2D -n 1024 -t 100 -x 8 -y 1

echo "16 MPI 1X16"
mpirun -np 16 ./cardiacsim2D -n 1024 -t 100 -x 1 -y 16

echo "16 MPI 2X8"
mpirun -np 16 ./cardiacsim2D -n 1024 -t 100 -x 2 -y 8

echo "16 MPI 4X4"
mpirun -np 16 ./cardiacsim2D -n 1024 -t 100 -x 4 -y 4

echo "16 MPI 8X2"
mpirun -np 16 ./cardiacsim2D -n 1024 -t 100 -x 8 -y 2

echo "16 MPI 16X1"
mpirun -np 16 ./cardiacsim2D -n 1024 -t 100 -x 16 -y 1

echo "32 MPI 1X32"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 1 -y 32

echo "32 MPI 2X16"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 2 -y 16

echo "32 MPI 4X8"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 4 -y 8

echo "32 MPI 8X4"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 8 -y 4

echo "32 MPI 16X2"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 16 -y 2

echo "32 MPI 32X1"
mpirun -np 32 ./cardiacsim2D -n 1024 -t 100 -x 32 -y 1
#....
echo "Finished with execution!"