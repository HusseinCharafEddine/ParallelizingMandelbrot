#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    int temp;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT); // Writing Width and Height
    fprintf(pgmimg, "255\n"); // Writing the maximum gray value
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            temp = image[i][j];
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    double start_time, end_time, elapsed_time, comm_time, total_elapsed_time = 0, total_comm_time = 0;
    int rank, size;
    int N = 10; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int k = 0; k < N; k++) {
        start_time = MPI_Wtime();

        int rows_per_process = HEIGHT / size;
        int start_row = rows_per_process * rank;
        int end_row = start_row + rows_per_process;
        int image[HEIGHT][WIDTH];

        if (rank == size - 1) {
            end_row = HEIGHT;
        }

        struct complex c;
        int *row = (int *) malloc(sizeof(int) * WIDTH);
        int *data = (int *) malloc(sizeof(int) * WIDTH * rows_per_process);

        for (int y = start_row; y < end_row; y++) {
            for (int x = 0; x < WIDTH; x++) {
                c.real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
                int iter = cal_pixel(c);
                row[x] = (iter == MAX_ITER) ? 0 : (iter % 256);
            }
            int row_index = (y - start_row) * WIDTH;
            for (int x = 0; x < WIDTH; x++) {
                data[row_index + x] = row[x];
            }
        }

        free(row);

        double start_comm_time = MPI_Wtime();
        MPI_Gather(data, WIDTH * rows_per_process, MPI_INT, image, WIDTH * rows_per_process, MPI_INT, 0,
                   MPI_COMM_WORLD);
        double end_comm_time = MPI_Wtime();
        comm_time = end_comm_time - start_comm_time;

        free(data);

        if (rank == 0) {
            save_pgm("static.pgm", image);
        }

        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        total_elapsed_time += elapsed_time;
        total_comm_time += comm_time;
    }

    if (rank == 0) {
        printf("The average execution time of %d trials is: %f ms\n", N, (total_elapsed_time / N) * 1000);
        printf("The average communication time of %d trials is: %f ms\n", N, (total_comm_time / N) * 1000);
    }

    MPI_Finalize();
    return 0;
}
