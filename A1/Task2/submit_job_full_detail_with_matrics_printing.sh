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

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo


echo "+++++++++++++++++ Start Test Time : $(date) ++++++++++++++++++++++++++++" >> ./output_serial_simple.txt
echo "+++++++++++++++++ Start Test Time : $(date) ++++++++++++++++++++++++++++" >> ./output_serial_optimized.txt
echo "+++++++++++++++++ Start Test Time : $(date) ++++++++++++++++++++++++++++" >> ./output_A.txt
echo "+++++++++++++++++ Start Test Time : $(date) ++++++++++++++++++++++++++++" >> ./output_B.txt
echo "+++++++++++++++++ Start Test Time : $(date) ++++++++++++++++++++++++++++" >> ./output_C.txt


for fname in 4x4_hard_3.csv
do
    printf "<<Serial Simple>>" >> ./output_serial_simple.txt
    ./sudoku_solver 16 $fname >> ./output_serial_simple.txt
    printf "<<Serial Optimized>>" >> ./output_serial_optimized.txt
    ./sudoku_solver_optimized 16 $fname >> ./output_serial_optimized.txt
done


printf "\<<Parallel Time>>" >> ./output_A.txt
printf "\<<Parallel Time>>" >> ./output_B.txt
printf "\<<Parallel Time>>" >> ./output_C.txt
for fname in 4x4_hard_3.csv
do
	for ((i=1; i<17; i=i*2));
		do 

			export OMP_NUM_THREADS=$i 
			export KMP_AFFINITY=verbose,granularity=fine,compact

			echo "Threads = $i" >> ./output_A.txt
            echo "Threads = $i" >> ./output_B.txt
            echo "Threads = $i" >> ./output_C.txt
			
			# ./fname_blurring_parallel $fname >> ./=======.txt
            ./part2_A 16 $fname >> ./output_A.txt
            ./part2_B 16 $fname >> ./output_B.txt
            ./part2_C 16 $fname >> ./output_C.txt
	done	
done

printf "\n\n<<Scatter Vs Compact>>\n\n" >> ./output_A.txt
printf "\n\n<<Scatter Vs Compact>>\n\n" >> ./output_B.txt
printf "\n\n<<Scatter Vs Compact>>\n\n" >> ./output_C.txt
for fname in 4x4_hard_3.csv
do
    
	
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,compact 
	echo "Compact" >> ./output_A.txt
	echo "Compact" >> ./output_B.txt
	echo "Compact" >> ./output_C.txt	
	./part2_A 16 $fname >> ./output_A.txt
    ./part2_B 16 $fname >> ./output_B.txt
    ./part2_C 16 $fname >> ./output_C.txt


	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,scatter
	echo  "Scattered" >> ./output_A.txt
	echo  "Scattered" >> ./output_B.txt
	echo  "Scattered" >> ./output_C.txt
	./part2_A 16 $fname >> ./output_A.txt
    ./part2_B 16 $fname >> ./output_B.txt
    ./part2_C 16 $fname >> ./output_C.txt
done


echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output_serial_simple.txt
echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output_serial_optimized.txt
echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output_A.txt
echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output_B.txt
echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output_C.txt



