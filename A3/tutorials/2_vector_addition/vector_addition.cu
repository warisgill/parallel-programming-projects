#include <stdio.h>

// version 1: parallel blocks
// __global__ void add(int *a, int *b, int * c){
//     int i = blockIdx.x;
//     c[i] = a[i] + b[i];
// }

// version 2: parallel threads
// __global__ void add(int *a, int *b, int * c){
//     int i = threadIdx.x;
//     c[i] = a[i] + b[i];
// }


// // version 3: Combining Threads and Blocks
// // blockDim.x  is the number of threads per block 
// __global__ void add(int *a, int *b, int * c){
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     c[i] = a[i] + b[i];
// }

// version 4: Handling Arbitrary Vector Sizes
/*
    1. blockDim.x  is the number of threads per block 
    2. Passing n so that index does not exceeds the array size
*/
__global__ void add(int *a, int *b, int * c, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


#define N 10
#define THREADS_PER_BLOCK 3 

int main(void){
    int *a, *b, *c;
    int *d_a,  *d_b, *d_c;
    int size = N * sizeof(int);
    int i = 0; 

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);


    // Allocate Memory on Device
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    // initialize variables
    // random_ints(a,N);
    // random_ints(b,N);

    for(i= 0; i<N; i++){
        a[i] = rand() % 40;
        b[i] = a[i];
    }


    // copy Data to GPU (device)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launching add() Kernel on GPU with N parallel blocks (Version 1)
    //add<<<N,1>>> (d_a, d_b, d_c);

    // Launching add() Kernel on GPU with N parallel threads (Version 2)
    //add<<<1,N>>>(d_a, d_b, d_c);

    // // Launching add() Kernel on GPU with N parallel blocks and threads (Version 3)
    // int number_of_blocks =  N/THREADS_PER_BLOCK;
    // add<<< number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Launching add() Kernel arbitrary vector sizes (Version 4)
    /*
        Why we are subtracting 1?
    */
    int number_of_blocks =  (N + THREADS_PER_BLOCK -1 )/THREADS_PER_BLOCK;
    add<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c,N);

    // Copy result to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


    // Print Orignal Vectors and Result
   
    printf("\na = ");
    for(i =0;i<N; i++ ){
        printf("%d, ", a[i]);
    }
    printf("\nb = ");
    for(i =0;i<N; i++ ){
        printf("%d, ", b[i] );
    }
    printf("\nc =  ");
    for(i =0;i<N; i++ ){
        printf("%d, ", c[i] );
    }
    printf("\n");


    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}


