/*
    This tutorial will explains why we need to bother with threads in cuda?
    Threads seem unnecessary
         They add a level of complexity
         What do we gain?
    
    Unlike parallel blocks, threads have mechanisms to efficiently:
         Communicate
         Synchronize
    
    To look closer, we need a stencil computation example.
*/


#include <stdio.h>

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*2)

__global__ void stencil_1d_simple(int *in, int *out) {
    // compute this thread's global index
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x + RADIUS;

    // int alpha = 1; 
    // int beta = 1; 

    if(i < NUM_ELEMENTS + RADIUS ){

        /* FIX ME #1 */

        out[i] = in[i-3] +in[i-2] + in[i-1] + in[i] + in[i+1] + in[i+2] + in[i+3];

    }
}


/*
    This function will not work properly because of race condition. 
    For instance, suppose that thread 15 reads the halo before thread 0 has fetched it from the global memory 
    and continue its computation which will leads to the race condition.  
*/

__global__ void stencil_1d_with_race_condition(int *in, int* out){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int global_i = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    int local_i = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[local_i] = in[global_i];
    if(threadIdx.x < RADIUS){
        temp[local_i - RADIUS] = in [global_i - RADIUS];
        temp[local_i + BLOCK_SIZE] = in[global_i + BLOCK_SIZE];
    }

    // Apply the stencil compution 
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset ++){
        result += temp[local_i + offset];
    }

    // Store the result
    out[global_i] = result;

}


/*
     void __syncthreads();
    
    Synchronizes all threads within a block
         Used to prevent RAW / WAR / WAW hazards
    
    All threads must reach the barrier
         In conditional code, the condition must be uniform across the block
*/

__global__ void stencil_1d_improved(int *in, int *out) {
    
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS]; /* FIXME #2*/

    int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + RADIUS; /* FIXME #3*/
    int lindex = threadIdx.x  + RADIUS; /* FIXME #4 */

    // Read input elements into shared memory
    temp[lindex] = in[gindex];

    //Load ghost cells (halos)
    if (threadIdx.x < RADIUS) {
       /* FIXME #5 */
       temp[lindex - RADIUS] =  in[gindex - RADIUS];
       temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // Make sure all threads get to this point before proceeding!
       /* FIXME #6 */	     
    __syncthreads(); // This will syncronize threads in a block

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];

    // Store the result
    out[gindex] = result;
}



int main() {
    unsigned int i;
    int N = NUM_ELEMENTS + 2 * RADIUS; 
    int h_in[N], h_out[N];
    int *d_in, *d_out;

    // Initialize host data
    for( i = 0; i < (N); ++i )
        h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7

    // Allocate space on the device
    cudaMalloc( &d_in,  N * sizeof(int)) ;
    cudaMalloc( &d_out, N * sizeof(int)) ;

    // Copy input data to device
    cudaMemcpy( d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice) ;

    //stencil_1d_simple<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);
    stencil_1d_improved<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_in, d_out);

    cudaMemcpy( h_out, d_out, N *  sizeof(int), cudaMemcpyDeviceToHost) ;

    // Verify every out value is 7
    for( i = RADIUS; i < NUM_ELEMENTS+RADIUS; ++i )
        if (h_out[i] != RADIUS*2+1)
        {
        printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
        break;
        }

    if (i == NUM_ELEMENTS+RADIUS)
        printf("SUCCESS!\n");

    // Free out memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
