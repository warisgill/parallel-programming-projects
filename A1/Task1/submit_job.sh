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


echo "+++++++++++++++++ Start Test Time : $(date)  <Schedule is static 32 job is halted> ++++++++++++++++++++++++++++" >> ./output.txt
printf "<<Serial Time>>\n" >> ./output.txt
for image in coffee.png cilek.png
do
	./image_blurring $image >> ./output.txt
done 


printf "\n<<Parallel Time>>" >> ./output.txt
for image in coffee.png cilek.png
do
	printf "\n\n" >> ./output.txt
	for ((i=1; i<17; i=i*2));
		do 
			echo "Threads = $i" >> ./output.txt
			export OMP_NUM_THREADS=$i 
			export KMP_AFFINITY=verbose,granularity=fine,compact
			./image_blurring_parallel $image >> ./output.txt
		done	
done

printf "\n<<Scatter Vs Compact>> \n" >> ./output.txt
for image in coffee.png cilek.png
do
	echo -e "Compact" >> ./output.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,compact 
	./image_blurring_parallel $image >> ./output.txt


	echo -e "Scattered">> ./output.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,scatter
	./image_blurring_parallel $image >> ./output.txt
done

echo "++++++++++++++++++++ End Test Time: $(date) ++++++++++++++++++++++++++++" >> ./output.txt
printf "\n" >> ./output.txt
