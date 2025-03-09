/*
Let a n by m matrix (A(n,m) in Fortran) be created on processor 0. For example, it might be read in from memory, or it might have been computed. Assume that there are four processors, and processor zero will send a part of this matrix to the other processors. Processor 1 gets A(i,j) for i=n/2+1,...,n, and j=1,...,m/2. Processor 2 gets A(i,j) for i=1,...,n/2 and j=m/2+1,...,m, and processor 3 gets A(i,j) for i=n/2+1,...,n and j=m/2,...,m . This is just a two-dimensional decomposition of A across four processors.
Use MPI_Scatterv to send the data from process 0 to all other processes (including process 0). For simplicity, choose n = m = 8.
*/

#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    double A[8][8], alocal[4][4];
    int i, j, r, rank, size;
    MPI_Datatype stype, t[2], vtype;
    MPI_Aint displs[2];
    int blklen[2];
    int sendcount[4], sdispls[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        fprintf(stderr, "This program requires exactly four processors\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        /* Initialize the matrix. Note that C has row-major storage */
        for (j = 0; j < 8; j++)
            for (i = 0; i < 8; i++)
                A[i][j] = 1.0 + i / 10.0 + j / 100.0;

        /* Form the vector type for the submatrix */
        MPI_Type_vector(4, 4, 8, MPI_DOUBLE, &vtype);
        /* Set an UB so that we can place this in the matrix */
        t[0] = vtype;
        t[1] = MPI_UB;
        displs[0] = 0;
        displs[1] = 4 * sizeof(double);
        blklen[0] = 1;
        blklen[1] = 1;
        MPI_Type_struct(2, blklen, displs, t, &stype);
        MPI_Type_commit(&stype);

        /* Setup the Scatter values for the send buffer */
        sendcount[0] = 1;
        sendcount[1] = 1;
        sendcount[2] = 1;
        sendcount[3] = 1;
        sdispls[0] = 0;
        sdispls[1] = 1;
        sdispls[2] = 8;
        sdispls[3] = 9;
        MPI_Scatterv(&A[0][0], sendcount, sdispls, stype,
                     &alocal[0][0], 4*4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv((void *)0, (void *)0, (void *)0, MPI_DATATYPE_NULL,
                     &alocal[0][0], 4*4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Everyone can now print their local matrix */
    for (r = 0; r < size; r++) {
        if (rank == r) {
            printf("Output for process %d\n", r);
            for (j = 0; j < 4; j++) {
                for (i = 0; i < 4; i++)
                    printf("%.2f ", alocal[i][j]);
                printf("\n");
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}