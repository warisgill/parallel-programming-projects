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

printf "\n\n\n" >>./simple.txt
printf "\n\n\n" >>./simple_optimized.txt
printf "\n\n\n" >>./output_A.txt
printf "\n\n\n" >>./output_B.txt
printf "\n\n\n" >>./output_C.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./simple.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./simple_optimized.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./output_A.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./output_B.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./output_C.txt

for fname in 4x4_hard_3.csv
do
    printf "\n\n\n Serial Simple  \n\n\n" >> ./simple.txt
    ./sudoku_solver 16 $fname >> ./simple.txt

    ./sudoku_solver_optimized 16 $fname >> ./simple_optimized.txt
    printf "\n\n\n Serial Optimized  \n\n\n" >> ./simple_optimized.txt
done


printf "\n\n\n Performance under Thread 1,4,8,16, if possible 32 \n\n\n" >> ./output_A.txt
printf "\n\n\n Performance under Thread 1,4,8,16, if possible 32 \n\n\n" >> ./output_B.txt
printf "\n\n\n Performance under Thread 1,4,8,16, if possible 32 \n\n\n" >> ./output_C.txt
for fname in 4x4_hard_3.csv
do
	for ((i=1; i<17; i=i*2));
		do 
			echo "<<Threads = $i threads" >> ./output_A.txt
            echo "<<Threads = $i threads" >> ./output_B.txt
            echo "<<Threads = $i threads" >> ./output_C.txt
			export OMP_NUM_THREADS=$i 
			export KMP_AFFINITY=verbose,granularity=fine,compact
			# ./fname_blurring_parallel $fname >> ./=======.txt
            ./part2_A 16 $fname >> ./output_A.txt
            ./part2_B 16 $fname >> ./output_B.txt
            ./part2_C 16 $fname >> ./output_C.txt
	done	
done

printf "\n\n\n Thread Binding Test \n\n\n" >> ./output_A.txt
printf "\n\n\n Thread Binding Test \n\n\n" >> ./output_B.txt
printf "\n\n\n Thread Binding Test \n\n\n" >> ./output_C.txt
for fname in 4x4_hard_3.csv
do
	
    printf "\n\n\n"
    echo -e "<<\n ****** Parallel version with 16 threads Compact **************************>>" >> ./output_A.txt
	echo -e "<<\n ****** Parallel version with 16 threads Compact **************************>>" >> ./output_B.txt
	echo -e "<<\n ****** Parallel version with 16 threads Compact **************************>>" >> ./output_C.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,compact 
	./part2_A 16 $fname >> ./output_A.txt
    ./part2_B 16 $fname >> ./output_B.txt
    ./part2_C 16 $fname >> ./output_C.txt

    printf "\n\n\n"
	echo -e "<<\n ****** Parallel version with 16 threads Scatter **************************>>" >> ./output_A.txt
	echo -e "<<\n ****** Parallel version with 16 threads Scatter **************************>>" >> ./output_B.txt
	echo -e "<<\n ****** Parallel version with 16 threads Scatter **************************>>" >> ./output_C.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,scatter
	./part2_A 16 $fname >> ./output_A.txt
    ./part2_B 16 $fname >> ./output_B.txt
    ./part2_C 16 $fname >> ./output_C.txt
done


echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./simple.txt
echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./simple_optimized.txt
echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./output_A.txt
echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./output_B.txt
echo "            +++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./output_C.txt

