#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# -= Resources =-
#
#SBATCH --job-name=image-blurring-jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --output=image-blurring-jobs.out

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

printf "\n\n\n" >>./OUTPUT_without_printing.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./OUTPUT_without_printing.txt

for fname in 4x4_hard_3.csv
do
    printf "\n\n\n Serial Simple  \n\n\n" >> ./OUTPUT_without_printing.txt
    ./sudoku_solver 16 $fname >> ./OUTPUT_without_printing.txt

    ./sudoku_solver_optimized 16 $fname >> ./OUTPUT_without_printing.txt
    printf "\n\n\n Serial Optimized  \n\n\n" >> ./OUTPUT_without_printing.txt
done


printf "\n\n\n Performance under Thread 1,4,8,16, if possible 32 \n\n\n" >> ./OUTPUT_without_printing.txt

for fname in 4x4_hard_3.csv
do
	for ((i=1; i<17; i=i*2));
		do 
            echo "<<Threads = $i threads" >> ./OUTPUT_without_printing.txt
			export OMP_NUM_THREADS=$i 
			export KMP_AFFINITY=verbose,granularity=fine,compact
			# ./fname_blurring_parallel $fname >> ./=======.txt
            ./part2_A 16 $fname >> ./OUTPUT_without_printing.txt
            ./part2_B 16 $fname >> ./OUTPUT_without_printing.txt
            ./part2_C 16 $fname >> ./OUTPUT_without_printing.txt
	done	
done

printf "\n\n\n Thread Binding Test \n\n\n" >> ./OUTPUT_without_printing.txt
for fname in 4x4_hard_3.csv
do
	
    printf "\n\n\n"
	echo -e "<<\n ****** Parallel version with 16 threads Compact **************************>>" >> ./OUTPUT_without_printing.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,compact 
	./part2_A 16 $fname >> ./OUTPUT_without_printing.txt
    ./part2_B 16 $fname >> ./OUTPUT_without_printing.txt
    ./part2_C 16 $fname >> ./OUTPUT_without_printing.txt

    printf "\n\n\n"
	echo -e "<<\n ****** Parallel version with 16 threads Scatter **************************>>" >> ./OUTPUT_without_printing.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,scatter
	./part2_A 16 $fname >> ./OUTPUT_without_printing.txt
    ./part2_B 16 $fname >> ./OUTPUT_without_printing.txt
    ./part2_C 16 $fname >> ./OUTPUT_without_printing.txt
done


echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./OUTPUT_without_printing.txt


