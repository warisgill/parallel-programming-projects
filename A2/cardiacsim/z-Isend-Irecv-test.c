#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
double **alloc2D(int m, int n)
{
    double **E;
    int nx = n; 
    int ny = m;
    E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
    assert(E);
    int j;
    for (j = 0; j < ny; j++)
        E[j] = (double *)(E + ny) + j * nx;
    return (E);
}
int main()
{
    int my_rank, comm_sz;
    int N = 20;
    double** send_buffer=alloc2D(N,N);
    double** recv_buff; 
    
    int p = 1;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Request req;
    MPI_Status stat;

    int local_m = N/comm_sz;
    int local_n = N ;

    recv_buff = alloc2D(local_m,local_n); 
    int tag = 0;
    if(my_rank == 0){
        int i;
        int j;
        for(i = 0; i<N; i++){
            
            for (j = 0; j<N;j ++){
                send_buffer[i][j] = i;
            }
            // send_buffer[i] = i; 
        }
        int start_from = local_n * local_m;
        printf("sshh\n");
        printf("sshh\n");
        printf("sshh\n");
        printf("sshh\n");
        for(i = 1; i<comm_sz; i++){
            printf("Hello how are you\n");
            printf("Hello how are you\n");
            printf("Hello how are you\n");
            printf("Hello how are you\n");
            
            MPI_Isend(&send_buffer[i*local_m][0],local_n*local_m,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&req);
        }
    } else {
        MPI_Irecv(&recv_buff[0][0],local_n*local_m,MPI_DOUBLE,0,tag,MPI_COMM_WORLD,&req);
    }

    MPI_Wait(&req,&stat);

    // MPI_Barrier(MPI_COMM_WORLD);

    if(my_rank != 0){
        printf("MY_ID: %d\n",my_rank);
        int i;
        int j;
        for(i= 0; i< local_m; i++){
            for(j = 0; j< local_n; j++){
                printf("%f ",recv_buff[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    

    free(send_buffer);
    free(recv_buff);
    MPI_Finalize();
    return 0;
}
