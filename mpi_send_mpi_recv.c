/*
Write code to copy divide the array x into equal-sized strips and to copy the adjacent edges to the neighboring processors. Assume that x is maxn by maxn, and that maxn is evenly divided by the number of processors. For simplicity, You may assume a fixed size array and a fixed (or minimum) number of processors.
For this exercise, use MPI_Send and MPI_Recv.
*/

#include <stdio.h>
#include "mpi.h"
#define maxn 12
int main(int argc, char* argv[]) {
    int rank, value, size, errcnt, toterr, i, j;
    MPI_Status status;
    double x[12][12];
    double xlocal[(12/4)+2][12];
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if (size != 4) MPI_Abort( MPI_COMM_WORLD, 1 );
    for (i=1; i<=maxn/size; i++)
        for (j=0; j<maxn; j++)
            xlocal[i][j] = rank;
    for (j=0; j<maxn; j++) {
        xlocal[0][j] = -1;
        xlocal[maxn/size+1][j] = -1;
    }
    if (rank < size - 1)
    MPI_Send( xlocal[maxn/size], maxn, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD );
    if (rank > 0)
    MPI_Recv( xlocal[0], maxn, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
    &status );
    if (rank > 0)
        MPI_Send( xlocal[1], maxn, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD );
    if (rank < size - 1)
        MPI_Recv( xlocal[maxn/size+1], maxn, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status );
    errcnt = 0;
    for (i=1; i<=maxn/size; i++)
        for (j=0; j<maxn; j++)
            if (xlocal[i][j] != rank) errcnt++;
    for (j=0; j<maxn; j++) {
        if (xlocal[0][j] != rank - 1) errcnt++;
        if (rank < size-1 && xlocal[maxn/size+1][j] != rank + 1) errcnt++;
    }
    MPI_Reduce( &errcnt, &toterr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
    if (rank == 0) {
        if (toterr)
            printf( "! found %d errors\n", toterr );
        else
            printf( "No errors\n" );
    }
    MPI_Finalize( );
    return 0;
}