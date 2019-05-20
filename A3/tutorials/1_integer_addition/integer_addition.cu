#include <stdio.h>

__global__ void add (int* a, int* b, int * c){

    *c = *a + *b;
    printf("\n value = %d \n", *c );
}

int main(){

    int a, b,c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    a = 1;
    b = 1;

    // Allocate Space on Device (GPU)
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size); 

    // Copy Data to Device 
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    // Launch the kernel  on GPU
    add<<<4,4>>>(d_a,d_b,d_c);  // this kernel will be executed 16 times but the output value will remain the same (2)

    // Copy Results back to Host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    


    printf("Result =  %d \n", c);

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

