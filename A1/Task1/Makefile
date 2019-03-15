# icc: image_blurring.c stb_image.h stb_image_write.h
# 	icc image_blurring.c -o image_blurring -qopenmp -lm

# gcc: image_blurring.c stb_image.h stb_image_write.h
# 	gcc -o image_blurring image_blurring.c -fopenmp -lm

icc: task1.c stb_image.h stb_image_write.h
	icc task1.c -o task1 -qopenmp -lm

gcc: task1.c stb_image.h stb_image_write.h
	gcc -o task1 task1.c -fopenmp -lm

