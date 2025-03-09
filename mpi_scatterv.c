/*
Use the Jacobi linear solver from the previous examples, as well as the data output from the "Collecting data" example.
You may want to use these MPI routines in your solution:
MPI_Scatterv
*/

#include <stdio.h>
#include <math.h>
#include "mpi.h"

/* This example handles a 12 x 12 mesh, on 4 processors only. */
#define maxn 12

int main(int argc, char **argv) {
    int rank, value, size, errcnt, toterr, i, j, itcnt;
    int i_first, i_last;
    MPI_Status status;
    double diffnorm, gdiffnorm;
    double xlocal[(12/4)+2][12];
    double xnew[(12/3)+2][12];
    double x[maxn][maxn];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0) {
        FILE *fp;
        fp = fopen("in.dat", "r");
        if (!fp) MPI_Abort(MPI_COMM_WORLD, 1);
        for (i = maxn - 1; i >= 0; i--) {
            for (j = 0; j < maxn; j++) {
                fscanf(fp, "%lf", &x[i][j]);
            }
            fscanf(fp, "\n");
        }
    }

    MPI_Scatter(x[0], maxn * (maxn / size), MPI_DOUBLE,
                xlocal[1], maxn * (maxn / size), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    i_first = 1;
    i_last = maxn / size;
    if (rank == 0) i_first++;
    if (rank == size - 1) i_last--;

    itcnt = 0;
    do {
        if (rank < size - 1)
            MPI_Send(xlocal[maxn / size], maxn, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        if (rank > 0)
            MPI_Recv(xlocal[0], maxn, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);

        if (rank > 0)
            MPI_Send(xlocal[1], maxn, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
        if (rank < size - 1)
            MPI_Recv(xlocal[maxn / size + 1], maxn, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);

        itcnt++;
        diffnorm = 0.0;
        for (i = i_first; i <= i_last; i++)
            for (j = 1; j < maxn - 1; j++) {
                xnew[i][j] = (xlocal[i][j + 1] + xlocal[i][j - 1] +
                              xlocal[i + 1][j] + xlocal[i - 1][j]) / 4.0;
                diffnorm += (xnew[i][j] - xlocal[i][j]) * (xnew[i][j] - xlocal[i][j]);
            }

        for (i = i_first; i <= i_last; i++)
            for (j = 1; j < maxn - 1; j++)
                xlocal[i][j] = xnew[i][j];

        MPI_Allreduce(&diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gdiffnorm = sqrt(gdiffnorm);
        if (rank == 0) printf("At iteration %d, diff is %e\n", itcnt, gdiffnorm);
    } while (gdiffnorm > 1.0e-2 && itcnt < 100);

    MPI_Gather(xlocal[1], maxn * (maxn / size), MPI_DOUBLE,
               x, maxn * (maxn / size), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final solution is\n");
        for (i = maxn - 1; i >= 0; i--) {
            for (j = 0; j < maxn; j++)
                printf("%f ", x[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}