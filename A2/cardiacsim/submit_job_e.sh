#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=cardiac-sim
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --constraint=e52695v4,36cpu
#SBATCH --time=30:00
#SBATCH --output=cardiacsim-%j.out

################################################################################
################################################################################

## Load openmpi version 3.0.0
echo "Loading openmpi module ..."
module load openmpi/3.0.0

## Load GCC-7.3.0
echo "Loading GCC module ..."
module load gcc/7.3.0

echo "PART E"

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

# Different MPI configurations
echo "Without OPENMP"

echo "1 MPI"
mpirun -np 1 ./cardiacsim2D -n 256 -t 10000

echo "2 MPI"
mpirun -np 2 ./cardiacsim2D -n 512 -t 5000

echo "4 MPI"
mpirun -np 4 ./cardiacsim2D -n 1024 -t 2500

echo "8 MPI"
mpirun -np 8 ./cardiacsim2D -n 2048 -t 1250

echo "16 MPI"
mpirun -np 16 ./cardiacsim2D -n 4096 -t 625

echo "32 MPI"
mpirun -np 32 ./cardiacsim2D -n 8192 -t 312
#....
echo "Finished with execution!"