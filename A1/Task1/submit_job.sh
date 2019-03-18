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






# # echo "===============================================================================">> ./output.txt
# printf "===========================Time at start %s =========================================\n" "$now">> ./output.txt
# echo "Running Job...!" >> ./output.txt
# echo "Running compiled binary..." >> ./output.txt

# echo "< old output.txt removed >" >> ./output.txt
# rm output.txt



#parallel version
# echo "< Parallel version with 32 threads >"
# export OMP_NUM_THREADS=32 
# export KMP_AFFINITY=verbose,granularity=fine,compact
# ./image_blurring_parallel coffee.png

# echo "Parallel version with 4 threads"
# export OMP_NUM_THREADS=16
# export KMP_AFFINITY=verbose,granularity=fine,compact
# ./image_blurring_parallel coffee.png

printf "\n\n" >> ./output.txt
printf "\n\n\n" >>./output.txt
echo "            +++++++++++++++++ Start Time: $(date) ++++++++++++++++++++++++++++" >> ./output.txt


printf "\n\n\n Serial Code  \n\n\n" >> ./output.txt
for image in coffee.png cilek.png
do
	echo "< **********Image: $image Serial Version  ************** >" >> ./output.txt
	# echo $image
	./image_blurring $image >> ./output.txt
done 

printf "\n\n\n Performance under Thread 1,4,8,16, if possible 32 \n\n\n" >> ./output.txt

for image in coffee.png cilek.png
do
	printf "\n\n\n" >> ./output.txt
	for ((i=1; i<17; i=i*2));
		do 
			echo "<<Threads = $i threads" >> ./output.txt
			export OMP_NUM_THREADS=$i 
			export KMP_AFFINITY=verbose,granularity=fine,compact
			./image_blurring_parallel $image >> ./output.txt
		done	
done

printf "\n\n\n Thread Binding Test \n\n\n" >> ./output.txt
for image in coffee.png cilek.png
do
	echo -e "<<\n ******Image: $image Parallel version with 16 threads Compact **************************>>" >> ./output.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,compact 
	./image_blurring_parallel $image >> ./output.txt


	echo -e "<<\n *******Image: $image Parallel version with 16 threads Scattered  ****************************>>">> ./output.txt
	export OMP_NUM_THREADS=16
	export KMP_AFFINITY=verbose,granularity=fine,scatter
	./image_blurring_parallel $image >> ./output.txt
done


echo "            ++++++++++++++++++++ End Time: $(date) ++++++++++++++++++++++++++++" >> ./output.txt
printf "\n\n\n" >> ./output.txt

 